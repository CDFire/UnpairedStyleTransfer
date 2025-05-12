import cv2
import os

def generate_edge_map(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    counter = 0
    for filename in os.listdir(input_folder):
        counter += 1
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, threshold1=30, threshold2=80)
        if img is None:
            print(f"Warning: Image '{filename}' is empty or could not be loaded. Skipping.")
            continue
        cv2.imwrite(os.path.join(output_folder, filename), edges)
        if counter % 10000 == 0:
            print(counter)

generate_edge_map('/Users/cole/Downloads/r', '/Users/cole/Downloads/r/edge_maps')
