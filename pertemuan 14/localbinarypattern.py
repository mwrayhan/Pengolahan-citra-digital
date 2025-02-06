import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Baca citra dalam greyscale
image = cv2.imread('bunga.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan Local Binary Pattern (LBP)
radius = 1
n_point = 8 * radius
lbp = local_binary_pattern(image, n_point, radius,method='uniform')

# Normalisasikan hasil LBP agar sesuai dengan rentang [0, 255]
lbp_normalized = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))

# Tampilkan hasil LBP
cv2.imshow('Local Binary Pattern', lbp_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()