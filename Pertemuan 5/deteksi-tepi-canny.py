import cv2

# Membaca gambar dalam grayscale
image = cv2.imread('image5.jpg', 0)

# Menerapkan deteksi tepi Canny
edges = cv2.Canny(image, 100, 200)

# Menampilkan hasil
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()