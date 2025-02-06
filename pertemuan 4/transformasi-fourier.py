import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar dalam grayscale
image = cv2.imread('image4.jpg', 0)

# Melakukan FFT
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Menghitung magnitude spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Menampilkan gambar asli dan magnitude spectrum
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('MagnitudeSpectrum')
plt.show()