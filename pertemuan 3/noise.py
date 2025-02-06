import cv2
import numpy as np
from matplotlib import pyplot as plt

# Fungsi untuk menampilkan gambar
def display_image(image, title, position):
    plt.subplot(3, 2, position)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Load a noisy image
image_path = 'image3.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Pastikan gambar memiliki noise

if image is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # Display original noisy image
    plt.figure(figsize=(15, 10))
    display_image(image, "Original Noisy Image", 1)

    # 1. Noise Reduction Filters

    # Apply Mean filter
    mean_filtered = cv2.blur(image, (5, 5))
    display_image(mean_filtered, "Mean Filtered Image", 2)

    # Apply Median filter
    median_filtered = cv2.medianBlur(image, 5)
    display_image(median_filtered, "Median Filtered Image", 3)

    # Apply Gaussian filter
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)
    display_image(gaussian_filtered, "Gaussian Filtered Image", 4)

    # 2. Histogram Equalization for Contrast Enhancement
    equalized_image = cv2.equalizeHist(image)
    display_image(equalized_image, "Histogram Equalized Image", 5)

    # 3. Geometric Transformations: Rotation and Scaling

    # Define rotation matrix
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = 45  # Rotate by 45 degrees
    scale = 1.2  # Scale by 120%
    
    # Get rotation matrix and perform rotation and scaling
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_scaled_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    display_image(rotated_scaled_image, "Rotated & Scaled Image", 6)

    # Display all images
    plt.tight_layout()
    plt.show()