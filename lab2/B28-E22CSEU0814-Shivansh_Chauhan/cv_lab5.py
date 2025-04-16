

import cv2
import numpy as np
import os
from PIL import Image

image_path = '/WhatsApp Image 2025-02-05 at 11.10.54_bf36262c.jpg'  
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please provide a valid image path.")

original_size = os.path.getsize(image_path)

lossless_path = 'compressed_lossless.png'
cv2.imwrite(lossless_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 2])
lossless_size = os.path.getsize(lossless_path)

lossy_path = 'compressed_lossy_70.jpg'
cv2.imwrite(lossy_path, image, [cv2.IMWRITE_JPEG_QUALITY, 70])
lossy_size_70 = os.path.getsize(lossy_path)

webp_path = 'compressed_lossy_70.webp'
cv2.imwrite(webp_path, image, [cv2.IMWRITE_WEBP_QUALITY, 70])
webp_size_70 = os.path.getsize(webp_path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
U, S, Vt = np.linalg.svd(image_gray, full_matrices=False)
k = 50 
compressed_image = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
pca_path = 'compressed_pca.jpg'
cv2.imwrite(pca_path, compressed_image)
pca_size = os.path.getsize(pca_path)

print(f"Original Image Size: {original_size / 1024:.2f} KB")
print(f"Lossless PNG Size: {lossless_size / 1024:.2f} KB")
print(f"Lossy JPEG Size (Quality 70): {lossy_size_70 / 1024:.2f} KB")
print(f"Lossy WebP Size (Quality 70): {webp_size_70 / 1024:.2f} KB")
print(f"PCA-Based Compression Size: {pca_size / 1024:.2f} KB")

if lossless_size > original_size:
    print("Warning: Lossless PNG compression increased the file size. Consider using a different format.")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
x_train_cifar = x_train_cifar.astype("float32") / 255.0
x_test_cifar = x_test_cifar.astype("float32") / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train_cifar = keras.utils.to_categorical(y_train_cifar, num_classes)
y_test_cifar = keras.utils.to_categorical(y_test_cifar, num_classes)

def create_cnn(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_mnist = create_cnn((28, 28, 1), num_classes)
history_mnist = model_mnist.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))

model_cifar = create_cnn((32, 32, 3), num_classes)
history_cifar = model_cifar.fit(x_train_cifar, y_train_cifar, epochs=50, batch_size=64, validation_data=(x_test_cifar, y_test_cifar))

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')
    print(f'AUC Score: {auc_score:.4f}')

evaluate_model(model_mnist, x_test, y_test)
evaluate_model(model_cifar, x_test_cifar, y_test_cifar)