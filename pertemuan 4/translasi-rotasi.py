import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('image4.jpg')

# Mendapatkan dimensi gambar
(h, w) = image.shape[:2]

# Translasi
M_translation = np.float32([[1, 0, 50], [0, 1, 100]])
translated_image = cv2.warpAffine(image, M_translation, (w, h))

# Rotasi 90 derajat
center = (w // 2, h // 2)
M_rotation = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(image, M_rotation, (w, h))

# Menampilkan hasil
cv2.imshow('Translated Image', translated_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()