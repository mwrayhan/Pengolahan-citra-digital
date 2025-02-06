import cv2

# Membaca gambar dalam grayscale
image = cv2.imread('image5.jpg', 0)

# Menerapkan adaptive thresholding
adaptive_tresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Menampilkan hasil
cv2.imshow('Adaptive Thresholding', adaptive_tresh)
cv2.waitKey(0)
cv2.destroyAllWindows()