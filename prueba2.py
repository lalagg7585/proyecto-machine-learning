import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar el dataset MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Visualizar algunas imágenes del dataset MNIST
fig, axes = plt.subplots(1, 10, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.set_axis_off()
    ax.imshow(X_train[i], cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Target: {y_train[i]}')
plt.show()