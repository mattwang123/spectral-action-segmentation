import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import ot

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

def cluster_frames_spectral(frames, n_clusters=3):
    """Cluster frames using Spectral Clustering."""
    X = np.array([f.flatten() for f in frames])
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        assign_labels='kmeans',
        random_state=42,
        n_neighbors=10
    )
    return spectral.fit_predict(X)

def compute_sliced_wasserstein_distance(img1, img2, n_projections=50):
    """Approximate Wasserstein distance using Sliced Wasserstein."""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    img1_flat = img1_flat / np.sum(img1_flat)
    img2_flat = img2_flat / np.sum(img2_flat)

    d = len(img1_flat)
    distances = []

    for _ in range(n_projections):
        proj = np.random.randn(d)
        proj /= np.linalg.norm(proj)

        proj1 = np.dot(img1_flat, proj)
        proj2 = np.dot(img2_flat, proj)

        distances.append(np.abs(proj1 - proj2))

    return np.mean(distances)


def cluster_frames_wasserstein(frames, n_clusters=3):
    """Cluster frames based on Wasserstein distances."""
    n_frames = len(frames)
    D = np.zeros((n_frames, n_frames))

    print("Computing pairwise Wasserstein distances...")
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            d = compute_sliced_wasserstein_distance(frames[i], frames[j])
            D[i, j] = d
            D[j, i] = d 

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'  
    )
    labels = clustering.fit_predict(D)
    return labels

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

if __name__ == "__main__":
    folder = os.path.join("preprocessed_frames", "denis_jump")
    frames = load_saved_frames(folder_path=folder, resize_dim=(64, 64))

    n_clusters = 3
    labels = cluster_frames_wasserstein(frames, n_clusters=n_clusters)

    plot_cluster_assignments(labels, method="Wasserstein")
    show_clustered_images(frames, labels, n_clusters=n_clusters, samples_per_cluster=6, method="Wasserstein")
    print("Wasserstein Clustering + Visualization complete.")