import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('image4.jpg')

# Titik sebelum transformasi
points1 = np.float32([[50, 50], [200, 50], [50, 200]])

# Titik setelah transformasi
points2 = np.float32([[10, 100], [200, 50], [100, 250]])

# Matriks transformasi affine
M_affine = cv2.getAffineTransform(points1, points2)

# Mendapatkan dimensi gambar
w, h = image.shape[:2]

# Melakukan transformasi affine
affine_transformed_image = cv2.warpAffine(image, M_affine, (w, h))

# Menampilkan hasil
cv2.imshow('Affine Transformed Image', affine_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()