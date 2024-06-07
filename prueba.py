import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2

# Cargar el dataset de lenguaje de señas
X_load = np.load('/kaggle/input/sign-language-digits-dataset/X.npy')
y_load = np.load('/kaggle/input/sign-language-digits-dataset/Y.npy')

# Preprocesar el dataset
img_size = 64
n_samples_signs = X_load.shape[0]
X_signs = X_load.reshape((n_samples_signs, -1))
y_signs = np.argmax(y_load, axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_signs, X_test_signs, y_train_signs, y_test_signs = train_test_split(X_signs, y_signs, test_size=0.2, shuffle=True)

# Entrenar el modelo SVM para las señales con las manos
clf_signs = SVC(gamma=0.001)
clf_signs.fit(X_train_signs, y_train_signs)

# Probar el modelo de señales con las manos
predicted_signs = clf_signs.predict(X_test_signs)
print(f"Accuracy for sign language: {accuracy_score(y_test_signs, predicted_signs)}")

from sklearn import datasets

# Cargar un conjunto de datos de dígitos
digits = datasets.load_digits()
X_digits = digits.images
y_digits = digits.target

# Preprocesar las imágenes (convertir a 1D)
n_samples_digits = len(X_digits)
X_digits = X_digits.reshape((n_samples_digits, -1))

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, test_size=0.5, shuffle=False)

# Entrenar el modelo SVM para los dígitos escritos
clf_digits = SVC(gamma=0.001)
clf_digits.fit(X_train_digits, y_train_digits)

# Probar el modelo de dígitos escritos
predicted_digits = clf_digits.predict(X_test_digits)
print(f"Accuracy for written digits: {accuracy_score(y_test_digits, predicted_digits)}")

# Acceso a la cámara y predicción en tiempo real
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para obtener una imagen binaria
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # Filtrar contornos pequeños
            # Extraer el área del contorno y redimensionar al tamaño esperado por los modelos
            roi = gray[y:y+h, x:x+w]
            roi_resized_digits = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)
            roi_resized_signs = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_AREA)

            # Preprocesar para los modelos
            roi_digits = roi_resized_digits.reshape((1, -1))
            roi_signs = roi_resized_signs.reshape((1, -1))

            # Predicción con ambos modelos
            number_digit = clf_digits.predict(roi_digits)
            number_sign = clf_signs.predict(roi_signs)
            
            # Dibujar el rectángulo y el número predicho
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Digit: {int(number_digit)}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f'Sign: {int(number_sign)}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Mostrar el resultado en una ventana
    cv2.imshow("Frame", frame)
    
    # Salir del bucle al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
