import cv2

# Baca citra grayscale
image = cv2.imread('image11.jpg', cv2.IMREAD_GRAYSCALE)

# Deteksi tepi menggunakan Canny
edges = cv2.Canny(image, 100, 200)

# Tampilkan hasil
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllwindows()