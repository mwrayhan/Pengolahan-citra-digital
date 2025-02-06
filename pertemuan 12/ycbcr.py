import cv2

# Baca citra dalam format RGB
image = cv2.imread('image12.jpg')

# Konversi citra dari RGB ke YCbCr
ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Ekstrak channel Y (luminance)
Y_channel = ycbcr_image[:,:,0]

# Tampilkan channel Y
cv2.imshow('Y Channel', Y_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()