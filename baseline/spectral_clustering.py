import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

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
        n_neighbors=10 # we can prob tune this 
    )
    labels = spectral.fit_predict(X)
    return labels

def plot_cluster_assignments(labels, save_path=None):
    """Plot the cluster assignments over time."""
    plt.figure(figsize=(10, 3))
    plt.plot(labels, marker='o')
    plt.xlabel("Frame Index")
    plt.ylabel("Cluster Label")
    plt.title("Spectral Clustering Over Time")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def show_clustered_images(frames, labels, n_clusters=3, samples_per_cluster=6, save_path=None):
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
    plt.suptitle("Clustered Frame Samples (Spectral Clustering)", fontsize=14)
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Input folder of preprocessed frames
    action_name = "shahar_pjump"
    folder_path = os.path.join("..", "preprocessed_frames", action_name)
    frames = load_saved_frames(folder_path=folder_path, resize_dim=(64, 64))

    # Output folder for saving results
    output_base = "../cluster_output"
    output_folder = os.path.join(output_base, "spectral_cluster_output", action_name)
    os.makedirs(output_folder, exist_ok=True)

    n_clusters = 3

    labels = cluster_frames_spectral(frames, n_clusters=n_clusters)
    plot_cluster_assignments(labels, save_path=os.path.join(output_folder, "cluster_assignments.png"))
    show_clustered_images(frames, labels, n_clusters=n_clusters, samples_per_cluster=6, save_path=os.path.join(output_folder, "clustered_images.png"))
    print("Spectral Clustering + Visualization complete.")
