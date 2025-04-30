import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import cv2
import os

def load_saved_frames(folder_path, resize_dim=(64, 64)):
    """Load saved grayscale PNG frames and normalize to [0, 1]."""
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
            n_clusters: Number of action stages to segment
            n_neighbors: For adaptive affinity matrix
            forecast_window: Sliding window size for prediction smoothing
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.forecast_window = forecast_window
        self.transition_matrix = None
        self.cluster_centers_ = None

    def _wasserstein_distance(self, X, Y):
        """Sliced Wasserstein Distance approximation between 2 frames"""
        # Implement your actual Wasserstein distance here
        return pairwise_distances(X.reshape(1, -1), Y.reshape(1, -1), metric='euclidean')[0][0]

    def _compute_affinity(self, frames):
        """Step 3: Affinity matrix with adaptive local scaling"""
        n = len(frames)
        D = np.zeros((n, n))
        
        # Pairwise Wasserstein distances
        for i in range(n):
            for j in range(i+1, n):
                D[i,j] = self._wasserstein_distance(frames[i], frames[j])
                D[j,i] = D[i,j]
        
        # Adaptive bandwidth
        sigma = np.zeros(n)
        for i in range(n):
            dists = np.sort(D[i])
            sigma[i] = dists[self.n_neighbors]
        
        # Affinity matrix
        W = np.exp(-D**2 / (sigma[:, None] * sigma[None, :]))
        np.fill_diagonal(W, 0)
        return W

    def fit(self, frames):
        """Steps 1-5: Full segmentation pipeline"""
        # 1. Compute affinity matrix
        W = self._compute_affinity(frames)
        
        # 2. Spectral clustering
        sc = SpectralClustering(n_clusters=self.n_clusters, 
                              affinity='precomputed',
                              assign_labels='kmeans')
        labels = sc.fit_predict(W)
        labels = np.unique(labels)
        # 3. Learn transition matrix
        self._learn_transition_matrix(labels)
        
        # 4. Store cluster centers (for visualization)
        self.cluster_centers_ = self._compute_cluster_centers(frames, labels)
        
        return labels

    def _learn_transition_matrix(self, labels):
        """Step 6: Build transition probability matrix"""
        K = self.n_clusters
        self.transition_matrix = np.zeros((K, K)) + 1e-6  # Laplace smoothing
        
        for i in range(len(labels)-1):
            current, next_ = labels[i], labels[i+1]
            self.transition_matrix[current, next_] += 1
        
        # Normalize rows
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

    def predict_next_stage(self, recent_labels):
        """Step 7: Predict next action stage"""
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

# ======================
# Example Usage
# ======================
if __name__ == "__main__":
    # 0. Simulate some video frames (replace with real data)
    num_frames = 100

    folder = os.path.join("preprocessed_frames", "denis_jump")
    frames = load_saved_frames(folder_path=folder, resize_dim=(64, 64))
    
    # 1. Initialize and fit
    segmenter = ActionSegmenterWithForecasting(n_clusters=4)
    labels = segmenter.fit(frames)
    
    print("Learned Transition Matrix:")
    print(segmenter.transition_matrix)
    
    # 2. Simulate real-time prediction
    current_window = labels[:5]  # Last 5 observed stages
    next_stage, confidence = segmenter.predict_next_stage(current_window)
    print(f"Predicted next stage: {next_stage} (confidence: {confidence:.2f})")
    
    # 3. Visualize cluster centers (representative frames)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, segmenter.n_clusters)
    for k, center in enumerate(segmenter.cluster_centers_):
        axes[k].imshow(center, cmap='gray')
        axes[k].set_title(f'Stage {k}')
    plt.show()