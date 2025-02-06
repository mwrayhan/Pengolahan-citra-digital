import cv2
import numpy as np

# Baca citra dalam grayscale
image = cv2.imread('bunga.jpg', cv2.IMREAD_GRAYSCALE)
# Terapkan Harris Corner Detector

gray = np.float32(image)  # Konversi ke tipe float32
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Tingkatkan sudut yang terdeteksi
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Konversi ke BGR untuk pewarnaan
image_color[corners > 0.01 * corners.max()] = [0, 0, 255]  # Warna merah untuk sudut

# Tampilkan hasil deteksi sudut
cv2.imshow('Harris Corner Detection', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()