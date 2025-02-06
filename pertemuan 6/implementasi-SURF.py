import cv2

# Membaca gambar
image = cv2.imread('merah.jpeg')

# Periksa apakah gambar berhasil dibaca
if image is None:
    print("Error: Gambar tidak ditemukan.")
else:
    # Inisialisasi objek ORB (alternatif pengganti SURF)
    orb = cv2.ORB_create()

    # Mendeteksi keypoints dan komputasi deskriptor
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Menggambar keypoints di citra
    orb_image = cv2.drawKeypoints(image, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Tampilkan hasil
    cv2.imshow('ORB Features', orb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()