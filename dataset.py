import os
import csv
import random
from tqdm import tqdm

# Directories containing your images
frames_dir = "/Users/cole/Downloads/dataset/RenderFrames"
edge_dir = "/Users/cole/Downloads/dataset/Render/edge_maps"
depth_dir = "/Users/cole/Downloads/dataset/Render/depth_maps"
color_dir = "/Users/cole/Downloads/dataset/Render/color_maps"

# Output CSV file name
output_csv = "render_dataset_prev.csv"

# Get a sorted list of filenames from the frames folder
all_filenames = sorted(os.listdir(frames_dir))

# We want consecutive frames.
# If you want a subset of consecutive frames, choose a contiguous block.
num_samples = 10000  # desired number of pairs (which requires num_samples+1 frames)
if len(all_filenames) - 1 > num_samples:
    # Choose a random starting index such that we have enough consecutive frames
    start_idx = random.randint(0, len(all_filenames) - num_samples - 1)
    selected_filenames = all_filenames[start_idx : start_idx + num_samples + 1]
else:
    selected_filenames = all_filenames

# Create and write to the CSV file with a progress bar.
# The CSV will contain:
#   filename: current frame's filename,
#   prev_frame_path: path to the previous frame,
#   frame_path: current frame path,
#   edge_path, depth_path, and color_path for the current frame.
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "prev_frame_path", "frame_path", "edge_path", "depth_path", "color_path"])

    # Start at index 1 so that each current frame has a previous frame.
    for i in tqdm(range(1, len(selected_filenames)), desc="Processing consecutive frames"):
        prev_filename = selected_filenames[i - 1]
        current_filename = selected_filenames[i]

        # Construct full file paths
        frame_path = os.path.join(frames_dir, current_filename)
        prev_frame_path = os.path.join(frames_dir, prev_filename)
        edge_path = os.path.join(edge_dir, current_filename)
        depth_path = os.path.join(depth_dir, current_filename)
        color_path = os.path.join(color_dir, current_filename)

        # Check that the file exists in all folders before writing to CSV
        if (os.path.exists(frame_path) and os.path.exists(prev_frame_path) and
            os.path.exists(edge_path) and os.path.exists(depth_path) and os.path.exists(color_path)):
            writer.writerow([current_filename, prev_frame_path, frame_path, edge_path, depth_path, color_path])

print(f"Dataset CSV with consecutive frames created and saved to {output_csv}")