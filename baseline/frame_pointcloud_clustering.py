import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import ot

# -------------------------------
# Load frames
# -------------------------------
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

# -------------------------------
# Frame to Point Cloud
# -------------------------------
def frame_to_point_cloud(frame, add_gradient=False):
    """
    Convert a 2D grayscale frame into a (N, 3) or (N, 4) point cloud.
    (x, y, intensity [, gradient])
    """
    H, W = frame.shape
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    
    x_norm = x_coords / (W - 1)
    y_norm = y_coords / (H - 1)
    intensity = frame  # already normalized [0,1]

    if add_gradient:
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag /= grad_mag.max() + 1e-8  # Avoid divide by 0
        point_cloud = np.stack([x_norm.flatten(), y_norm.flatten(), intensity.flatten(), grad_mag.flatten()], axis=1)
    else:
        point_cloud = np.stack([x_norm.flatten(), y_norm.flatten(), intensity.flatten()], axis=1)

    return point_cloud  # shape (H*W, 3) or (H*W, 4)

# -------------------------------
# Spectral Clustering
# -------------------------------
def cluster_frames_spectral(point_clouds, n_clusters=3):
    """Cluster frames using Spectral Clustering."""
    # Flatten each point cloud to a single vector
    X = np.array([pc.flatten() for pc in point_clouds])
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        assign_labels='kmeans',
        random_state=42,
        n_neighbors=10
    )
    return spectral.fit_predict(X)

# -------------------------------
# Sliced Wasserstein Distance between Point Clouds
# -------------------------------
def compute_sliced_wasserstein_distance(pc1, pc2, n_projections=50):
    """Approximate Wasserstein distance between two point clouds."""
    d = pc1.shape[1]
    distances = []

    for _ in range(n_projections):
        proj = np.random.randn(d)
        proj /= np.linalg.norm(proj)

        proj1 = np.dot(pc1, proj)
        proj2 = np.dot(pc2, proj)

        proj1.sort()
        proj2.sort()

        # Compute 1D Wasserstein distance (L1 distance between sorted projections)
        distance = np.mean(np.abs(proj1 - proj2))
        distances.append(distance)

    return np.mean(distances)

def cluster_frames_wasserstein(point_clouds, n_clusters=3):
    """Cluster frames based on Wasserstein distances between point clouds."""
    n_frames = len(point_clouds)
    D = np.zeros((n_frames, n_frames))

    print("Computing pairwise Sliced Wasserstein distances between point clouds...")
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            d = compute_sliced_wasserstein_distance(point_clouds[i], point_clouds[j])
            D[i, j] = d
            D[j, i] = d

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',  # <- Use metric not affinity
        linkage='average'
    )
    labels = clustering.fit_predict(D)
    return labels

# -------------------------------
# Visualization
# -------------------------------
def plot_cluster_assignments(labels, method="Spectral"):
    """Plot the cluster assignments over time."""
    plt.figure(figsize=(10, 3))
    plt.plot(labels, marker='o')
    plt.xlabel("Frame Index")
    plt.ylabel("Cluster Label")
    plt.title(f"{method} Clustering Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_clustered_images(frames, labels, n_clusters=3, samples_per_cluster=6, method="Spectral"):
    """Display a grid of clustered images."""
    plt.figure(figsize=(samples_per_cluster * 2, n_clusters * 2.5))
    for c in range(n_clusters):
        indices = np.where(labels == c)[0][:samples_per_cluster]
        for j, idx in enumerate(indices):
            plt.subplot(n_clusters, samples_per_cluster, c * samples_per_cluster + j + 1)
            plt.imshow(frames[idx], cmap='gray')
            plt.axis('off')
            plt.title(f"C{c}", fontsize=8)
    plt.tight_layout()
    plt.suptitle(f"Clustered Frame Samples ({method} Clustering)", fontsize=14)
    plt.show()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    folder = os.path.join("preprocessed_frames", "denis_jump")
    frames = load_saved_frames(folder_path=folder, resize_dim=(64, 64))

    # Convert frames into structured point clouds
    point_clouds = [frame_to_point_cloud(f, add_gradient=True) for f in frames]
    print(f"Converted {len(point_clouds)} frames into point clouds.")

    n_clusters = 3
    method = "wasserstein"   # Options: 'spectral' or 'wasserstein'

    if method == "spectral":
        labels = cluster_frames_spectral(point_clouds, n_clusters=n_clusters)
    elif method == "wasserstein":
        labels = cluster_frames_wasserstein(point_clouds, n_clusters=n_clusters)
    else:
        raise ValueError("Unknown clustering method.")

    plot_cluster_assignments(labels, method=method.capitalize())
    show_clustered_images(frames, labels, n_clusters=n_clusters, samples_per_cluster=6, method=method.capitalize())
    print(f"{method.capitalize()} Clustering + Visualization complete.")
