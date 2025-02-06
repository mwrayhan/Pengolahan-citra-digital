import cv2

# Membaca gambar 
image = cv2.imread('merah.jpeg')

# Inisialisasi objek SIFT
sift = cv2.SIFT_create()

# Mendeteksi keypoints dan komputasi deskriptor 
keypoints, descriptors = sift.detectAndCompute (image, None)

# Menggambar keypoints di citra 
sift_image = cv2.drawKeypoints (image, keypoints, None)

#Menampilkan hasil
cv2.imshow('SIFT Features', sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()