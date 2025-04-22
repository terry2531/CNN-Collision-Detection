import cv2
import os
import numpy as np
from datetime import datetime

i = '093'

# Define folder paths
folder1 = fr'C:\Users\A\Desktop\video001\{i}'  # Replace with your first folder path
folder2 = fr'C:\Users\A\Desktop\new\{i}'       # Replace with your second folder path
output_base_folder = r'C:\Users\A\Desktop\002'  # Replace with your output folder path

# Get current timestamp for creating a unique output folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_folder = os.path.join(output_base_folder, f'{i}')

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all image filenames in both folders, sorted alphabetically
images1 = sorted(os.listdir(folder1))
images2 = sorted(os.listdir(folder2))

# Filter out non-image files and system hidden files (e.g., .DS_Store and files starting with ._)
images1 = [img for img in images1 if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and not img.startswith('._')]
images2 = [img for img in images2 if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and not img.startswith('._')]

# Determine the smaller number of images from both folders
num_images = min(len(images1), len(images2))

# Iterate over each pair of images
for i in range(num_images):
    # Build image paths
    img1_path = os.path.join(folder1, images1[i])
    img2_path = os.path.join(folder2, images2[i])

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Check if images are read successfully
    if img1 is None:
        print(f"Failed to read image: {img1_path}")
        continue  # Skip this image
    if img2 is None:
        print(f"Failed to read image: {img2_path}")
        continue  # Skip this image

    # Ensure both images have the same size
    if img1.shape != img2.shape:
        # Resize img2 to match img1's size
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Blend the two images
    combined_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    # Save the blended image
    output_path = os.path.join(output_folder, f'combined_{i+1}.png')
    cv2.imwrite(output_path, combined_img)

print(f'Processing complete. Blended images saved in: {output_folder}')
