# @title
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=Path)
parser.add_argument("--save_path", type=Path)
args = parser.parse_args()

def segment_main_cell(image_path, output_size=(256, 256)):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to create a binary image
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to separate touching cells
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == 0] = [255, 0, 0]

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which we assume to be the main cell
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    main_cell = image[y:y+h, x:x+w]
    resized_cell = cv2.resize(main_cell, output_size, interpolation=cv2.INTER_AREA)

    return resized_cell

# Path to the folder containing images
folder_path = args.folder_path
save_path = args.save_path

# Ensure the save path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        main_cell = segment_main_cell(image_path)

        # Save the main cell with the same name as the original image
        cell_filename = f"{os.path.splitext(filename)[0]}_main_cell.jpg"
        cell_path = os.path.join(save_path, cell_filename)
        cv2.imwrite(cell_path, main_cell)