import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

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
# Frame Difference Selection
# -------------------------------
def compute_frame_differences(frames):
    """Compute absolute frame-to-frame differences."""
    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i] - frames[i - 1])
        diffs.append(diff)
    print(f"Computed {len(diffs)} frame differences.")
    return diffs

def select_motion_keyframes(frames, top_k=40):
    """Select indices of frames with highest motion."""
    diffs = compute_frame_differences(frames)
    motion_strength = [np.sum(d) for d in diffs]
    ranked_indices = np.argsort(motion_strength)[-top_k:]  # top_k motion-heavy
    return sorted(ranked_indices)  # return in chronological order

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
    intensity = frame

    if add_gradient:
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag /= grad_mag.max() + 1e-8
        point_cloud = np.stack([x_norm.flatten(), y_norm.flatten(), intensity.flatten(), grad_mag.flatten()], axis=1)
    else:
        point_cloud = np.stack([x_norm.flatten(), y_norm.flatten(), intensity.flatten()], axis=1)

    return point_cloud

# -------------------------------
# Sliced Wasserstein Distance
# -------------------------------
def compute_sliced_wasserstein_distance(pc1, pc2, n_projections=50):
    d = pc1.shape[1]
    distances = []
    for _ in range(n_projections):
        proj = np.random.randn(d)
        proj /= np.linalg.norm(proj)
        proj1 = np.dot(pc1, proj)
        proj2 = np.dot(pc2, proj)
        proj1.sort()
        proj2.sort()
        distances.append(np.mean(np.abs(proj1 - proj2)))
    return np.mean(distances)

def cluster_frames_wasserstein(point_clouds, n_clusters=3):
    n_frames = len(point_clouds)
    D = np.zeros((n_frames, n_frames))
    print("Computing pairwise Sliced Wasserstein distances between point clouds...")
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            d = compute_sliced_wasserstein_distance(point_clouds[i], point_clouds[j])
            D[i, j] = D[j, i] = d
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return clustering.fit_predict(D)

# -------------------------------
# Visualization
# -------------------------------
def plot_cluster_assignments(labels, method="Wasserstein"):
    plt.figure(figsize=(10, 3))
    plt.plot(labels, marker='o')
    plt.xlabel("Frame Index")
    plt.ylabel("Cluster Label")
    plt.title(f"{method} Clustering Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_clustered_images(frames, labels, n_clusters=3, samples_per_cluster=6, method="Wasserstein"):
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
    folder = os.path.join("preprocessed_frames", "shahar_pjump")
    frames = load_saved_frames(folder_path=folder, resize_dim=(64, 64))

    # --- Select motion-heavy frames ---
    top_frame_indices = select_motion_keyframes(frames, top_k=40)
    selected_frames = [frames[i + 1] for i in top_frame_indices]  # +1 aligns with diff indexing

    # --- Convert selected frames to point clouds ---
    point_clouds = [frame_to_point_cloud(f, add_gradient=True) for f in selected_frames]
    print(f"Converted {len(point_clouds)} motion-based frames into point clouds.")

    # --- Cluster ---
    n_clusters = 3
    labels = cluster_frames_wasserstein(point_clouds, n_clusters=n_clusters)

    # --- Visualize ---
    plot_cluster_assignments(labels, method="Wasserstein")
    show_clustered_images(selected_frames, labels, n_clusters=n_clusters, samples_per_cluster=6, method="Wasserstein")
    print("Wasserstein Clustering on motion-based pointclouds complete.")
