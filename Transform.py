import cv2
import numpy as np
import glob
import os

# Read all image files and sort them by filename
image_files = sorted(glob.glob(r'C:\Users\A\PycharmProjects\PythonProject\project 002\video\093/*.png'))  # Replace with your image path

# Ensure there are at least two images
if len(image_files) < 2:
    print("Error: At least two images are required.")
    exit()

# Read the first frame
frame1 = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
h, w = frame1.shape

# Create folder to save optical flow frames
output_folder = r'C:\Users\A\Desktop\new\093'  # Replace with your desired output path
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all frames to compute optical flow
for i in range(1, len(image_files)):
    frame2 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Set saturation to max

    # Calculate magnitude and angle of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue represents direction
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value represents speed

    # Convert HSV to BGR for visualization
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Build the filename for saving the frame
    frame_filename = os.path.join(output_folder, f"optical_flow_{i - 1:04d}.jpg")

    # Save the optical flow frame
    cv2.imwrite(frame_filename, bgr)

    # Optionally display the current optical flow image
    cv2.imshow("Optical Flow", bgr)
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    # Update the previous frame
    frame1 = frame2.copy()

# Release resources
cv2.destroyAllWindows()

print(f"All optical flow frames have been saved to {output_folder}")
