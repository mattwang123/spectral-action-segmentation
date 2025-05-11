import cv2
import os

def generate_video_from_images(image_folder, output_video, fps=30):
    # Get list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Ensure images are in the correct order

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")

# Example usage
if __name__ == "__main__":
    image_folder = "path/to/your/image/folder"
    output_video = "output_video.mp4"
    fps = 30  # Frames per second
    generate_video_from_images(image_folder, output_video, fps)