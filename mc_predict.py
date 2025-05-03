import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from scipy.optimize import linear_sum_assignment
import cv2
import os
import matplotlib.pyplot as plt

def load_saved_frames(folder_path, resize_dim=(64, 64)):
    """Reads PNG frames that were already extracted from a video, 
        converts each to grayscale (one channel instead of three), 
        rescales them to a common spatial size (resize_dim), 
        normalizes the pixel range to [0, 1], and returns a list/array of float32 images."""
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    frames = []

    for fname in frame_files:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
            frames.append(img.astype(np.float32) / 255.0)

    print(f"Loaded {len(frames)} frames from '{folder_path}'")
    return frames

class ActionSegmenterWithForecasting:
    def __init__(self, n_clusters=5, n_neighbors=5, forecast_window=3):
        """
        Args:
            n_clusters: number of distinct action stages to segment
            n_neighbors: used later to set adaptive badnwidth per frame when converting pariwise distance into affinity matrix
            forecast_window: how many most recent labels are looked at when you later predict the next action stage
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.forecast_window = forecast_window
        self.transition_matrix = None   # This is the learned Markov Model
        self.cluster_centers_ = None    # This is the representative frames for each cluster
    

    def _wasserstein_distance(self, X, Y, n_projections=50):
        """Approximate the 2‑Wasserstein distance between two grayscale images
        using the Sliced‑Wasserstein technique.

        Each image is flattened into a 1‑D probability vector (values ≥0 that sum
        to 1).  We draw `n_projections` random directions on the unit hypersphere,
        project both distributions onto each direction, and take the average
        absolute difference of the resulting 1‑D marginals.  This Monte‑Carlo
        estimate is a fast, differentiable proxy for the full Wasserstein metric
        and is the same routine used in `baseline/spectral_wasserstein.py`.
        """
        # Flatten to vectors
        x = X.flatten().astype(np.float64)
        y = Y.flatten().astype(np.float64)

        # Normalise so each is a discrete probability distribution
        x_sum = x.sum();  y_sum = y.sum()
        if x_sum == 0:  x_sum = 1.0  # avoid divide‑by‑zero
        if y_sum == 0:  y_sum = 1.0
        x /= x_sum
        y /= y_sum

        d = x.size  # dimensionality

        # Draw random projection directions (each row has unit ℓ2‑norm)
        dirs = np.random.randn(n_projections, d)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        # Project both distributions
        proj_x = dirs @ x  # shape (n_projections,)
        proj_y = dirs @ y

        # 1‑D Wasserstein between projected distributions reduces to |Δ| here
        return float(np.mean(np.abs(proj_x - proj_y)))

    def _compute_affinity(self, frames):
        """Build the affinity matrix with adaptive local scaling.
            IMPORTANT THEORY2: Affinity matrix captures the similarity between frames using the Wasserstein distance."""
        n = len(frames)
        D = np.zeros((n, n)) # So this is the pairwise frame to frame distance matrix, notice it is symmetric
        
        # Pairwise Wasserstein distances
        for i in range(n):
            for j in range(i+1, n):
                D[i,j] = self._wasserstein_distance(frames[i], frames[j])
                D[j,i] = D[i,j] # symmetric !
        
        # Adaptive bandwidth sigma, for each frame i, take distance to k nearest neighbors. If slow change, then small, if jittery, large.
        sigma = np.zeros(n)
        # Avoid out-of-bounds for neighbor index
        k = min(self.n_neighbors, n - 1)
        for i in range(n):
            dists = np.sort(D[i])
            sigma[i] = dists[k]
        # Prevent division by zero
        sigma[sigma == 0] = np.finfo(float).eps
        
        # Important! Build the affinity matrix W from the distance matrix D
        W = np.exp(-D**2 / (sigma[:, None] * sigma[None, :]))
        np.fill_diagonal(W, 0)
        return W

    def _learn_transition_matrix(self, labels):
        """IMPORTANT THEORY3! Learn Markov Transition Matrix
            Transition_matrix[i][j] = Probability of going from stage i to stage j in the next frame
        Args:
            labels: 1D array of cluster labels for each frame
        """
        K = self.n_clusters
        self.transition_matrix = np.zeros((K, K)) + 1e-6  # Laplace smoothing
        
        for i in range(len(labels)-1):
            current, next_ = labels[i], labels[i+1]
            self.transition_matrix[current, next_] += 1 # Increment count of transition from current to next stage
        
        # Normalize rows to make sure probabilities sum to 1
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)


    def predict_next_stage(self, recent_labels):
        """Predict next action stage"""
        # Use sliding window of recent labels
        window = recent_labels[-self.forecast_window:]
        
        # Count transitions in window
        counts = np.zeros(self.n_clusters)
        for i in range(len(window)-1):
            current, next_ = window[i], window[i+1]
            counts += self.transition_matrix[current]
        
        # Weighted prediction
        predicted = np.argmax(counts)
        confidence = counts[predicted] / counts.sum()
        
        return predicted, confidence

    def _compute_cluster_centers(self, frames, labels):
        """For visualization: Find representative frames per cluster"""
        centers = []
        for k in range(self.n_clusters):
            cluster_frames = [frames[i] for i, lbl in enumerate(labels) if lbl == k]
            # Use frame closest to median Wasserstein distance
            center_idx = np.argmin([
                np.median([self._wasserstein_distance(f, f2) for f2 in cluster_frames])
                for f in cluster_frames
            ])
            centers.append(cluster_frames[center_idx])
        return centers
    

    def fit(self, frames, labels=None):
        """Fit the model
        Args:
            frames: List/array of video frames (grayscale, normalised).
            labels: Optional externally computed cluster labels (one per frame). If provided, the internal spectral clustering step is skipped and these labels are used directly.
        """
        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape[0] != len(frames):
                raise ValueError("Length of labels must match number of frames")
        else:
            # Model fit step 1: Compute affinity matrix
            W = self._compute_affinity(frames)
            # Model fit step 2: Spectral clustering, uses the affnity matrix W, theory is mainly computing top K eignevectors of normalized Lapacian. Whatever that means lol.
            sc = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='precomputed',
                assign_labels='kmeans'
            )
            labels = sc.fit_predict(W) # This is the cluster labels for each frame, so it is a 1D array of length n_frames.
        
        # Model fit step 3: Learn transition matrix
        self._learn_transition_matrix(labels)
        
        # Model fit step 4: Store cluster centers (for visualization)
        self.cluster_centers_ = self._compute_cluster_centers(frames, labels)
        
        return labels


# -------------------------------------------------------
# Example usage WOW!
# -------------------------------------------------------
if __name__ == "__main__":
    folder = os.path.join("preprocessed_frames", "shahar_jack")
    frames = load_saved_frames(folder_path=folder, resize_dim=(64, 64))
    
    # 1. Initialize and fit
    segmenter = ActionSegmenterWithForecasting(n_clusters=4)
    labels = segmenter.fit(frames)
    
    print("Learned Transition Matrix:")
    print(segmenter.transition_matrix)
    
    # 2. Simulate real-time prediction
    current_window = labels[:5]  # Last 5 observed stages
    next_stage, confidence = segmenter.predict_next_stage(current_window)
    print(f"Last observed stages: {current_window}")
    print(f"Predicted next stage: {next_stage} (confidence: {confidence:.2f})")
    
    # 3. Visualize cluster centers (representative frames)
    fig, axes = plt.subplots(1, segmenter.n_clusters)
    for k, center in enumerate(segmenter.cluster_centers_):
        axes[k].imshow(center, cmap='gray')
        axes[k].set_title(f'Stage {k}')
    plt.show()