import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Leer el CSV
data = pd.read_csv('heart.csv')

# Asignar características (features) y variable objetivo
X = data.iloc[:, :-1].values  # características
y = data.iloc[:, -1].values   # variable objetivo

# Agregar una columna de unos para el término independiente
X = np.c_[np.ones(X.shape[0]), X]

# Dividir el conjunto de datos en entrenamiento y prueba (80% - 20%)
def train_test_split(X, y, test_size=0.2):
    split_idx = int(len(y) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definir la función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función para entrenar el modelo de regresión logística
def train_logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    weights = np.zeros(X.shape[1])  # Inicializar pesos
    for _ in range(epochs):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        error = y - predictions
        gradient = np.dot(X.T, error)
        weights += learning_rate * gradient
    return weights

# Entrenar el modelo
weights = train_logistic_regression(X_train, y_train)

# Función para predecir usando los pesos aprendidos
def predict(X, weights):
    z = np.dot(X, weights)
    return np.round(sigmoid(z))

# Realizar predicciones en el conjunto de prueba
y_pred = predict(X_test, weights)

# Calcular métricas de evaluación
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

def recall(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])

def precision(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix[1, 1] / (matrix[0, 1] + matrix[1, 1])

def f1_score(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Calcular métricas de evaluación
# acc = accuracy(y_test, y_pred)
# rec = recall(y_test, y_pred)
# prec = precision(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# print("Accuracy:", acc)
# print("Recall:", rec)
# print("Precision:", prec)
# print("F1 Score:", f1)
# print("Confusion Matrix:")
# print(cm)
print("Mean Squared Error:", mse)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Matriz de Confusión')
plt.colorbar()

classes = ['0', '1']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Anotar los valores en la matriz
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()