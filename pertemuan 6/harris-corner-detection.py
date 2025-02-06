import cv2 
import numpy as np

# Membaca gambar dalam grayscale 
image = cv2.imread('merah.jpeg', 0)

# Konversi gambar ke tipe float32 
gray = np.float32(image)

# Menerapkan Harris Corner Detector 
dst = cv2.cornerHarris (gray, 2, 3, 0.04)

# Dilasi untuk menonjolkan sudut yang terdeteksi 
dst = cv2.dilate(dst, None)

# Konversi gambar grayscale ke BGR
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Thresholding untuk menandai sudut  
image_color[dst > 0.01 * dst.max()] = [0, 0, 255]

# Menampilkan hasil 
cv2.imshow('Harris Corners', image_color)
cv2.waitKey(0) 
cv2.destroyAllWindows()