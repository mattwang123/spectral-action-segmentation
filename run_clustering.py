import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

def load_saved_frames(folder_path, resize_dim=(64, 64)):
    """Load saved grayscale PNG frames and normalize to [0, 1]."""
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    frame_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])

    frames = []

    for fname in frame_files:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
            frames.append(img.astype(np.float32) / 255.0)

    print(f"Loaded {len(frames)} frames from '{folder_path}'")
    return frames


def compute_frame_differences(frames):
    """Compute absolute differences between consecutive frames."""
    diffs = [None]  # No difference for the first frame
    for i in range(1, len(frames)):
        diff = np.abs(frames[i] - frames[i - 1])
        diffs.append(diff)
    return diffs


def frame_to_point_cloud(frame):
    """Convert a grayscale frame into a point cloud of (x, y, intensity, gradient)"""
    H, W = frame.shape
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    x_norm = x_coords / (W - 1)
    y_norm = y_coords / (H - 1)
    intensity = frame  # already normalized [0,1]

    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag /= grad_mag.max() + 1e-8  # Avoid divide by 0
    pc = np.stack([x_norm.flatten(), y_norm.flatten(), intensity.flatten(), grad_mag.flatten()], axis=1)

    return pc

def frame_to_point_cloud(frame, diff=None):
    """Convert a frame (and optional temporal diff) to a 5D point cloud."""
    H, W = frame.shape
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    x_norm = x_coords / (W - 1)
    y_norm = y_coords / (H - 1)
    intensity = frame

    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag /= grad_mag.max() + 1e-8

    if diff is None:
        diff = np.zeros_like(frame)

    pc = np.stack([
        x_norm.flatten(),
        y_norm.flatten(),
        intensity.flatten(),
        grad_mag.flatten(),
        diff.flatten()
    ], axis=1)

    return pc



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


def cluster_point_clouds(point_clouds, distance="euclidean", method="spectral", n_clusters=3):
    """Unified interface: cluster point clouds using specified distance + method."""
    if distance == "euclidean":
        X = np.array([pc.flatten() for pc in point_clouds])
        if method == "spectral":
            model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
            return model.fit_predict(X)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42)
            return model.fit_predict(X)

    elif distance == "wasserstein":
        n = len(point_clouds)
        D = np.zeros((n, n))
        print("🔁 Computing Wasserstein distance matrix...")
        for i in range(n):
            for j in range(i + 1, n):
                d = compute_sliced_wasserstein_distance(point_clouds[i], point_clouds[j])
                D[i, j] = D[j, i] = d
        if method == "spectral":
            model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
            return model.fit_predict(D)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
            return model.fit_predict(D)

    else:
        raise ValueError("Unsupported distance type. Use 'euclidean' or 'wasserstein'.")


def plot_cluster_assignments(labels, method="Clustering"):
    plt.figure(figsize=(10, 3))
    plt.plot(labels, marker='o')
    plt.xlabel("Frame Index")
    plt.ylabel("Cluster Label")
    plt.title(f"{method} Clustering Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_clustered_images(frames, labels, n_clusters=3, samples_per_cluster=20, method="Clustering"):
    plt.figure(figsize=(samples_per_cluster * 2, n_clusters * 2.5))
    for c in range(n_clusters):
        indices = np.where(labels == c)[0][:samples_per_cluster]
        for j, idx in enumerate(indices):
            plt.subplot(n_clusters, samples_per_cluster, c * samples_per_cluster + j + 1)
            plt.imshow(frames[idx], cmap='gray')
            plt.axis('off')
            plt.title(f"C{c}", fontsize=8)
    plt.tight_layout()
    plt.suptitle(f"Clustered Frame Samples ({method})", fontsize=14)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster video frames as point clouds using various methods.")
    parser.add_argument('--folder', type=str, default="preprocessed_frames/denis_jump", help="Input frame folder.")
    parser.add_argument('--clusters', type=int, default=3, help="Number of clusters.")
    parser.add_argument('--distance', type=str, choices=["euclidean", "wasserstein"], default="euclidean",
                        help="Distance metric to use.")
    parser.add_argument('--method', type=str, choices=["spectral", "kmeans"], default="kmeans",
                        help="Clustering method to use.")
    args = parser.parse_args()

    frames = load_saved_frames(folder_path=args.folder)
    diffs = compute_frame_differences(frames)
    pcs = [frame_to_point_cloud(f, diff=d) for f, d in zip(frames, diffs)]

    labels = cluster_point_clouds(pcs, distance=args.distance, method=args.method, n_clusters=args.clusters)

    title = f"{args.method.capitalize()}-{args.distance.capitalize()}"
    plot_cluster_assignments(labels, method=title)
    show_clustered_images(frames, labels, n_clusters=args.clusters, method=title)
    print(f"✅ {title} clustering complete.")