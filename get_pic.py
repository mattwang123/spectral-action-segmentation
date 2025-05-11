import cv2
import os

def custom_crop(frame):
    """Crop 1/5 from bottom, 1/5 from left, and 3/5 from right."""
    h, w, _ = frame.shape

    # Horizontal crop: keep from 1/5 to 2/5 (i.e., 20% to 40%)
    left = int(w * 0.2)
    right = int(w * 0.6)

    # Vertical crop: keep top 4/5
    top = int(h * 0.1)
    bottom = int(h * 0.8)

    return frame[top:bottom, left:right]

def extract_three_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_indices = [1, 14, 19]
    print(f"Extracting custom-cropped frames at indices: {frame_indices}")

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cropped = custom_crop(frame)
            out_path = os.path.join(output_dir, f"frame_{i+1}.png")
            cv2.imwrite(out_path, cropped)
            print(f"Saved {out_path}")
        else:
            print(f"❌ Failed to read frame at index {frame_idx}")

    cap.release()

# Example usage
extract_three_frames("data/shahar_jack.avi", "extracted_frames_cropped")
