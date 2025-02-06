import cv2
import numpy as np
from matplotlib import pyplot as plt
# Membaca gambar
image = cv2.imread('imageakusisi.jpg', cv2.IMREAD_GRAYSCALE)
# Menampilkan gambar
cv2.imshow('Original Image', image)
cv2.waitKey(0)
# Menampilkan histogram
plt.hist(image.ravel(), 256, [0,256])
plt.title('Histogram of Image')
plt.show()
# Menutup jendela OpenCV
cv2.destroyAllWindows()