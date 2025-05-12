import cv2
import numpy as np
import os
import concurrent.futures

def process_image(filename, input_folder, output_folder, num_colors):
    image_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Image '{filename}' is empty or could not be loaded. Skipping.")
        return False
    # Reshape and convert to float32 for K-means
    data = img.reshape((-1, 3)).astype(np.float32)
    # Apply K-means clustering
    _, labels, centers = cv2.kmeans(
        data, num_colors, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    cv2.imwrite(output_path, segmented_image)
    return True

def generate_color_map(input_folder, output_folder, num_colors=8):
    os.makedirs(output_folder, exist_ok=True)
    filenames = os.listdir(input_folder)
    total_images = len(filenames)
    processed_count = 0

    # Process images in parallel using a process pool
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Prepare tasks for all images
        futures = {
            executor.submit(process_image, filename, input_folder, output_folder, num_colors): filename
            for filename in filenames
        }
        # As each task completes, update progress
        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            if processed_count % 10000 == 0:
                print(f"Processed {processed_count} images out of {total_images}")
            # Optionally check if processing failed for an image
            if not future.result():
                filename = futures[future]
                print(f"Failed processing image: {filename}")

if __name__ == '__main__':
    generate_color_map('/Users/cole/Downloads/r', '/Users/cole/Downloads/r/color_maps')