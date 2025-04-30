import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

def compute_frame_differences(frames):
    """Compute absolute frame-to-frame differences."""
    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i] - frames[i - 1])
        diffs.append(diff)
    print(f"Computed {len(diffs)} frame differences.")
    return diffs

def cluster_frames_kmeans(frames, n_clusters=3):
    X = np.array([f.flatten() for f in frames])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X)

def plot_cluster_assignments(labels):
    """Plot the cluster assignments over time."""
    plt.figure(figsize=(10, 3))
    plt.plot(range(1, len(labels)+1), labels, marker='o')  # Start from frame 1
    plt.xlabel("Frame Index")
    plt.ylabel("Cluster Label")
    plt.title("K-Means Clustering Over Time (Frame Differences)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def show_clustered_images(frames, labels, n_clusters=3, samples_per_cluster=6):
    """Display original images from each cluster (using difference-based clustering)."""
    plt.figure(figsize=(samples_per_cluster * 2, n_clusters * 2.5))
    for c in range(n_clusters):
        indices = np.where(labels == c)[0][:samples_per_cluster]
        for j, idx in enumerate(indices):
            # +1 because diffs start from frame[1] - frame[0]
            img_idx = idx + 1
            plt.subplot(n_clusters, samples_per_cluster, c * samples_per_cluster + j + 1)
            plt.imshow(frames[img_idx], cmap='gray')
            plt.axis('off')
            plt.title(f"C{c}", fontsize=8)
    plt.tight_layout()
    plt.suptitle("Clustered Frame Samples (Motion-Based)", fontsize=14)
    plt.show()

if __name__ == "__main__":
    folder = os.path.join("preprocessed_frames", "denis_jump")
    frames = load_saved_frames(folder_path=folder, resize_dim=(64, 64))

    diff_frames = compute_frame_differences(frames)

    n_clusters = 2
    labels = cluster_frames_kmeans(diff_frames, n_clusters=n_clusters)
    plot_cluster_assignments(labels)
    show_clustered_images(frames, labels, n_clusters=n_clusters, samples_per_cluster=6)
    print("✅ Motion-based clustering complete.")
