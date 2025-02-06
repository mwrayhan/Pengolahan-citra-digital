import cv2

# Baca citra dalam greyscale
image = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan thresholding global
ret, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Tampilkan hasil segmentasi
cv2.imshow('Thresholded Image', otsu_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()