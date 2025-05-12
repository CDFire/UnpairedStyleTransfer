import cv2
import os


def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save each frame as an image file
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1
        if frame_count % 10000 == 0:
            print(frame_count)

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


# Example usage
video_path = '/Users/cole/Documents/Adobe/Premiere Pro/25.0/MushiEdit.mp4'
output_folder = '/Users/cole/Downloads/dataset/Frames'
extract_frames(video_path, output_folder)