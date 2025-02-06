import cv2
import numpy as np

# Baca citra dalam greyscale
image = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan thresholding global
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Tampilkan hasil segmentasi
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()