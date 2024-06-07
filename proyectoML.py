import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
from sklearn.decomposition import PCA
import tensorflow as tf
from get_dataset import get_dataset

# Función para cargar y preprocesar el dataset de lenguaje de señas
def load_sign_language_data():
    X_train, X_test, Y_train, Y_test = get_dataset()
    # Aplanar las imágenes
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

    # Convertir las etiquetas a unidimensionales si están en formato one-hot
    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    return  X_train, X_test, Y_train, Y_test

# Función para cargar y preprocesar el dataset MNIST
def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalizar y aplanar las imágenes
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0
    return X_train, X_test, y_train, y_test

X_train_signs, X_test_signs, y_train_signs, y_test_signs = load_sign_language_data()

#############################################################################
# Cargar los datos de MNIST
X_train_digits, X_test_digits, y_train_digits, y_test_digits = load_mnist_data()
"""
# Aplicar PCA sin reducir componentes para obtener la varianza explicada
pca = PCA()
pca.fit(X_train_digits)

# Calcular la varianza explicada acumulada
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(explained_variance >= 0.95) + 1

print(f"Cantidad mínima de componentes para capturar el 95% de la varianza: {n_components_95}")

#############################################################################

# Aplicar PCA sin reducir componentes para obtener la varianza explicada
pca2 = PCA()
pca2.fit(X_train_signs)

# Calcular la varianza explicada acumulada
explained_variance2 = np.cumsum(pca2.explained_variance_ratio_)

# Determinar el número de componentes necesarios para capturar al menos el 95% de la varianza
n_components_95_2 = np.argmax(explained_variance2 >= 0.95) + 1

print(f"Cantidad mínima de componentes para capturar el 95% de la varianza SIGN: {n_components_95_2}")

# Graficar la varianza explicada acumulada
plt.figure(figsize=(10, 6))
plt.plot(explained_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=n_components_95, color='r', linestyle='--')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada por el PCA en DIGITS')
plt.grid(True)
plt.show()

# Graficar la varianza explicada acumulada
plt.figure(figsize=(10, 6))
plt.plot(explained_variance2, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=n_components_95_2, color='r', linestyle='--')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada por el PCA en SIGN')
plt.grid(True)
plt.show()

# Aplicar PCA con el número óptimo de componentes para entrenamiento y prueba
pca_optimal = PCA(n_components=n_components_95)
X_train_digits_pca = pca_optimal.fit_transform(X_train_digits)
X_test_digits_pca = pca_optimal.transform(X_test_digits)

# También aplicar PCA a las señales con el mismo número de componentes para consistencia
pca_signs = PCA(n_components=n_components_95_2)
X_train_signs_pca = pca_signs.fit_transform(X_train_signs)
X_test_signs_pca = pca_signs.transform(X_test_signs)

"""

# Ajuste de hiperparámetros para kernel RBF y kernel polinomial
param_grid_rbf = {
    'C': [10],
    'gamma': ['scale']
}

param_grid_poly = {
    'C': [10],
    'gamma': ['scale']
}

# Ajustar el modelo SVM con kernel RBF
grid_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid_rbf, refit=True, cv=5, n_jobs=-1)
grid_rbf.fit(X_train_digits, y_train_digits)
print(f"Best parameters for RBF kernel: {grid_rbf.best_params_}")

# Ajustar el modelo SVM con kernel polinomial
grid_poly = GridSearchCV(SVC(kernel='poly'), param_grid_poly, refit=True, cv=5, n_jobs=-1)
grid_poly.fit(X_train_digits, y_train_digits)
print(f"Best parameters for polynomial kernel: {grid_poly.best_params_}")

# Ajustar el modelo SVM para señales con las manos utilizando el mejor kernel
#clf_signs = SVC(kernel='rbf', C=1, gamma='scale').fit(X_train_signs_pca, y_train_signs)

# Función para graficar los resultados del SVM
def plot_svm_decision_boundary(model, X, y, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.title(title)
    plt.show()

# Para la visualización, reducimos a 2 componentes
pca_visual = PCA(n_components=2)
X_train_digits_visual = pca_visual.fit_transform(X_train_digits)
X_test_digits_visual = pca_visual.transform(X_test_digits)
#X_train_signs_visual = pca_visual.fit_transform(X_train_signs)
#X_test_signs_visual = pca_visual.transform(X_test_signs)

# Graficar las fronteras de decisión para los modelos usando las 2 componentes principales
plot_svm_decision_boundary(grid_rbf.best_estimator_, X_test_digits_visual, y_test_digits, "SVM con Kernel RBF para Dígitos MNIST")
plot_svm_decision_boundary(grid_poly.best_estimator_, X_test_digits_visual, y_test_digits, "SVM con Kernel Polinomial para Dígitos MNIST")
#plot_svm_decision_boundary(clf_signs, X_test_signs_visual, y_test_signs, "SVM con Kernel RBF para Señales")

# Probar el modelo de señales con las manos
#predicted_signs = clf_signs.predict(X_test_signs_pca)
#print(f"Accuracy for sign language: {accuracy_score(y_test_signs, predicted_signs)}")

# Probar el modelo de dígitos escritos
predicted_digits = grid_rbf.best_estimator_.predict(X_test_digits)
print(f"Accuracy for written digits (RBF kernel): {accuracy_score(y_test_digits, predicted_digits)}")
predicted_digits_poly = grid_poly.best_estimator_.predict(X_test_digits_pca)
print(f"Accuracy for written digits (Polynomial kernel): {accuracy_score(y_test_digits, predicted_digits_poly)}")

# Función para preprocesar y predecir en tiempo real
def preprocess_and_predict(frame, clf_rbf, clf_poly, clf_signs, pca_digits, pca_signs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar un filtro Gaussiano para suavizar la imagen y reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtrar contornos pequeños y limitar el área de reconocimiento
        if 30 < w < 200 and 30 < h < 200:
            roi = gray[y:y+h, x:x+w]
            roi_resized_digits = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi_resized_signs = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
            roi_digits = roi_resized_digits.reshape((1, -1)) / 255.0
            roi_signs = roi_resized_signs.reshape((1, -1)) / 255.0

            # Aplicar PCA a las regiones de interés
            roi_digits_pca = pca_digits.transform(roi_digits)
            roi_signs_pca = pca_signs.transform(roi_signs)

            number_rbf = clf_rbf.predict(roi_digits_pca)
            number_poly = clf_poly.predict(roi_digits_pca)
            number_signs = clf_signs.predict(roi_signs_pca)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'RBF: {int(number_rbf)}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f'Poly: {int(number_poly)}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, f'Sign: {int(number_signs)}', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return frame

# Acceso a la cámara y predicción en tiempo real
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = preprocess_and_predict(frame, grid_rbf.best_estimator_, grid_poly.best_estimator_, clf_signs, pca_optimal, pca_signs)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
