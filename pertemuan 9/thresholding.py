import cv2
import numpy as np

# Baca citra grayscale
image = cv2.imread('image9.jpg', cv2.IMREAD_GRAYSCALE)

# Gloval thresholding
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Adaptive thresholding
thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Tampilkan hasil
cv2.imshow('Global Thresholding', thresh1)
cv2.imshow('Adaptive Thresholding', thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()