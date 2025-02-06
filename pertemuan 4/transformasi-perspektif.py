import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('image4.jpg')

# Mendefinisikan empat titik sudut citra asli
points1 = np.float32([[56, 65], [386, 52], [28, 387], [389, 390]])

# Mendefinisikan empat titik sudut baru
points2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# Mendapatkan matriks transformasi perspektif
M_perspective = cv2.getPerspectiveTransform(points1, points2)

# Melakukan transformasi perspektif
perspective_transformed_image = cv2.warpPerspective(image, M_perspective, (300, 300))

# Menampilkan hasil
cv2.imshow('Perspective Transformed Image', perspective_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()