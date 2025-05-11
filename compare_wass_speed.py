import os
import cv2
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from ot import emd2

# Load frames function

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

# Traditional Wasserstein Distance

def traditional_wasserstein(pc1, pc2):
    cost_matrix = np.abs(pc1[:, np.newaxis] - pc2[np.newaxis, :])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()

# Sliced Wasserstein Distance

def sliced_wasserstein(pc1, pc2, num_projections=50):
    d = pc1.shape[1]
    distances = []
    for _ in range(num_projections):
        proj = np.random.randn(d)
        proj /= np.linalg.norm(proj)
        proj1 = np.dot(pc1, proj)
        proj2 = np.dot(pc2, proj)
        proj1.sort()
        proj2.sort()
        distances.append(np.mean(np.abs(proj1 - proj2)))
    return np.mean(distances)

# POT Wasserstein Distance

def pot_wasserstein(pc1, pc2):
    n = len(pc1)
    M = np.abs(pc1[:, np.newaxis] - pc2[np.newaxis, :])
    p = np.ones(n) / n
    q = np.ones(n) / n
    return emd2(p, q, M)

# Comparison function

def compare_methods(frames):
    methods = ["Traditional", "Sliced", "POT"]
    times = {method: [] for method in methods}
    for i in range(len(frames) - 1):
        pc1, pc2 = frames[i].flatten(), frames[i+1].flatten()
        # Traditional
        start = time.time()
        traditional_wasserstein(pc1, pc2)
        times["Traditional"].append(time.time() - start)

        # Sliced
        start = time.time()
        sliced_wasserstein(pc1.reshape(-1, 1), pc2.reshape(-1, 1))
        times["Sliced"].append(time.time() - start)

        # POT
        start = time.time()
        pot_wasserstein(pc1, pc2)
        times["POT"].append(time.time() - start)

    print("\nComparison of Wasserstein Distance Methods:")
    print("Method       Average Time (s)")
    for method in methods:
        avg_time = np.mean(times[method])
        print(f"{method:<12} {avg_time:.6f}")

if __name__ == "__main__":
    folder = "preprocessed_frames/denis_jump"
    frames = load_saved_frames(folder)
    compare_methods(frames[:5])
