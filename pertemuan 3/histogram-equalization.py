import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar dalam grayscale
image = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

# Equalisasi histogram
equalized_image = cv2.equalizeHist(image)

# Menampilkan gambar asli dan hasil equalization
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Menampilkan histogram
plt.hist(equalized_image.ravel(), 256, [0,256])
plt.title('Histogram of Equalized Image')
plt.show()