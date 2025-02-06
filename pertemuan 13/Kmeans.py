import cv2
import numpy as np

# Baca citra
image = cv2.imread('image2.jpg')

# Ubah format citra ke dalam satu dimensi
Z = image.reshape((-1, 3))

# Ubah tipe data ke float32
Z = np.float32(Z)

# Tentukan kriteria dan jumlah cluster
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3 # Misalnya, 3 cluster

# Terapkan K-means clustering
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Konversi hasil kembali ke format citra
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((image.shape))

# Tampilkan hasil segmentasi
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()