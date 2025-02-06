import tensorflow as tf
from tensorflow.keras import layers, models

# Menggunakan dataset CIFAR-10 sebagai contoh
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalisasi data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Membangun model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Melatih model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluasi model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Akurasi CNN: {test_acc * 100:.2f}%')