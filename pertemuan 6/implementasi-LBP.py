import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Membaca gambar dalam grayscale
image = cv2.imread('merah.jpeg', cv2.IMREAD_GRAYSCALE)

# Menerapkan Local Binary Pattern
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# Normalisasi LBP ke rentang 0-255
lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)

# Mengubah tipe data menjadi uint8
lbp_image = lbp_normalized.astype(np.uint8)

# Menampilkan hasil
cv2.imshow('Local Binary Pattern', lbp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()