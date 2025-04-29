import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_video(video_path):
    """Load video and return capture object and FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def extract_frames(cap, original_fps, target_fps, resize_dim):
    """
    Frame extraction, resize, and grayscale normalization.
    """
    frame_interval = int(round(original_fps / target_fps))
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, resize_dim, interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            frames.append(normalized)

        frame_idx += 1

    cap.release()
    return frames

def save_frames(frames, output_dir):
    """Save = frames as images."""
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(frames):
        save_path = os.path.join(output_dir, f'frame_{idx:03d}.png')
        cv2.imwrite(save_path, (img * 255).astype(np.uint8))
    print(f"Saved {len(frames)} frames to {output_dir}")

def preprocess_video(video_path, target_fps=10, resize_dim=(64, 64), save_dir=None):
    """
    Full preprocessing pipeline:
    - Load video
    - Extract and preprocess frames
    - Optionally save frames
    - Optionally visualize a sample point cloud
    """
    cap, original_fps = load_video(video_path)
    frames = extract_frames(cap, original_fps, target_fps, resize_dim)

    if save_dir:
        save_frames(frames, save_dir)

    return frames

if __name__ == "__main__":
    video_path = 'data/denis_jump.avi'

    # Create a subfolder based on the video filename for output
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join('preprocessed_frames', base_name)

    print("Preprocessing data...")
    frames = preprocess_video(
        video_path=video_path,
        target_fps=10,
        resize_dim=(64, 64),
        save_dir=output_dir
    )

    print(f"Preprocessing complete. Total frames extracted: {len(frames)}")

    # Show a grid of the first 9 preprocessed frames
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(frames[i], cmap='gray')
        ax.set_title(f"Frame {i}")
        ax.axis('off')
    plt.suptitle("First 9 Preprocessed Frames", fontsize=14)
    plt.tight_layout()
    plt.show()