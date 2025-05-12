import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Define global variables for model and feature extractor
model = None
feature_extractor = None

# GPU/CPU selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model():
    """Initialize model and feature extractor for each process."""
    global model, feature_extractor
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.to(device).eval()


def process_image(task):
    """Process an individual image to generate its depth map."""
    input_path, output_path = task
    try:
        # Load and preprocess the image
        original_image = Image.open(input_path).convert("RGB")
        original_size = original_image.size

        inputs = feature_extractor(images=original_image, return_tensors="pt").to(device)

        # Predict depth map
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

        # Normalize and resize depth map
        depth_min = predicted_depth.min()
        depth_max = predicted_depth.max()
        normalized_depth = (predicted_depth - depth_min) / (depth_max - depth_min + 1e-8)
        resized_depth = cv2.resize(normalized_depth, original_size, interpolation=cv2.INTER_LINEAR)

        # Save depth map
        plt.imsave(output_path, resized_depth, cmap='viridis')
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_folder_parallel(input_folder, output_folder, max_workers=8):
    """Process all images in a folder using parallel processing."""
    os.makedirs(output_folder, exist_ok=True)

    # Prepare list of tasks
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    tasks = [(os.path.join(input_folder, f), os.path.join(output_folder, f)) for f in image_files]

    # Parallel processing
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_model) as executor:
        results = list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc="Processing Frames"))

    print(f"\n✅ Successfully processed {sum(results)}/{len(tasks)} images into: {output_folder}")


if __name__ == "__main__":
    # Ensure safe multiprocessing
    input_folder = '/Users/cole/Downloads/r'
    output_folder = '/Users/cole/Downloads/r/depth_maps'

    # Set the optimal number of workers based on your system
    process_folder_parallel(input_folder, output_folder, max_workers=8)