import cv2
# Membaca gambar
image = cv2.imread('imageshow.jpg')
# Menampilkan gambar
cv2.imshow('Display Window', image)
# Menunggu hingga ada input dari keyboard
cv2.waitKey(0)
# Menutup semua jendela
cv2.destroyAllWindows()