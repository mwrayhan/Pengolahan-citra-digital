from skimage.feature import graycomatrix,  graycoprops
import cv2

# Membaca gambar dalam greyscale
image = cv2.imread('merah.jpeg', 0)

# Menghitung GLCM
glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# Menghitung fitur tekstur dari GLCM
contrast = graycoprops(glcm, 'contrast')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]

print(f'Contrast: {contrast}, Energy: {energy}')