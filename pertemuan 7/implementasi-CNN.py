from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np

# Contoh data dummy (ganti dengan data asli)
X_train = np.random.rand(500, 64, 64, 3)  # 500 gambar 64x64 dengan 3 channel warna
y_train = np.random.randint(0, 10, 500)   # 500 label untuk 10 kelas
X_test = np.random.rand(100, 64, 64, 3)   # 100 gambar untuk pengujian
y_test = np.random.randint(0, 10, 100)    # 100 label pengujian

# Inisialisasi model CNN
model = Sequential()

# Tambahkan convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tambahkan convolutional layer lainnya
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model
model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32)

# Evaluasi model
loss, accuracy = model.evaluate(X_test, to_categorical(y_test))
print(f'Akurasi CNN: {accuracy * 100:.2f}%')