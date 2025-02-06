import cv2

# Baca citra dalam greyscale
image = cv2.imread('bunga.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan deteksi tepi dengan metode canny
edges = cv2.Canny(image, 100, 200)

# Tampilkan hasil deteksi tepi
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()