from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Contoh data dummy (ganti dengan data aslimu)
X = np.random.rand(100, 5)  # 100 sampel dengan 5 fitur
y = np.random.randint(0, 2, 100)  # 100 label (biner: 0 atau 1)

# Split data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Latih model menggunakan data pelatihan
knn.fit(X_train, y_train)

# Prediksi menggunakan data uji
y_pred = knn.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi KNN: {accuracy * 100:.2f}%')