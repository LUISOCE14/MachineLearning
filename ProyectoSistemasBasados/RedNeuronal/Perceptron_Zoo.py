import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Cargar los datos desde un archivo CSV
df = pd.read_csv('zoo.csv')
names = df["animal_name"]
class_types = df["class_type"].unique()
data = df.drop(["animal_name", "class_type"], axis=1)

# Division de los datos en entrenamiento (80%) y prueba (20%)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Convertir los datos a matrices NumPy
X_train = train_data.iloc[:, :-1].values  
y_train = train_data.iloc[:, -1].values  
X_test = test_data.iloc[:, :-1].values   
y_test = test_data.iloc[:, -1].values   


# Configuración de la arquitectura de la red neuronal
input_neurons = X_train.shape[1]  
hidden_neurons = 4  
output_neurons = 1  

# Pesos aleatorios para las conexiones entre las capas
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))

epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X_train, hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, output_weights)
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = y_train.reshape(-1, 1) - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Actualización de pesos
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    hidden_weights += X_train.T.dot(d_hidden_layer) * learning_rate

# Predicción en el conjunto de prueba
hidden_layer_input_test = np.dot(X_test, hidden_weights)
hidden_layer_output_test = sigmoid(hidden_layer_input_test)

output_layer_input_test = np.dot(hidden_layer_output_test, output_weights)
predicted_output_test = sigmoid(output_layer_input_test)

# Conversion de predicciones a valores binarios
predicted_output_test_binary = (predicted_output_test > 0.5).astype(int)

# Funciones para métricas de evaluación
def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def confusion_matrix(y_true, y_pred):
    true_pos = np.sum((y_true == 1) & (y_pred == 1))
    true_neg = np.sum((y_true == 0) & (y_pred == 0))
    false_pos = np.sum((y_true == 0) & (y_pred == 1))
    false_neg = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[true_neg, false_pos], [false_neg, true_pos]])

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    true_pos = cm[1][1]
    false_pos = cm[0][1]
    return true_pos / (true_pos + false_pos + 1e-10)

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    true_pos = cm[1][1]
    false_neg = cm[1][0]
    return true_pos / (true_pos + false_neg + 1e-10)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + 1e-10)

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

accuracy_value = accuracy(y_test, predicted_output_test_binary.flatten())
conf_matrix = confusion_matrix(y_test, predicted_output_test_binary.flatten())
recall_value = recall(y_test, predicted_output_test_binary.flatten())
f1_value = f1_score(y_test, predicted_output_test_binary.flatten())
mse_value = mean_squared_error(y_test, predicted_output_test.flatten())

print(f"Accuracy: {accuracy_value}")
#print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Recall: {recall_value}")
print(f"F1 Score: {f1_value}")
print(f"MSE: {mse_value}")

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Matriz de Confusión')
plt.colorbar()

classes = ['0', '1']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Anotar los valores en la matriz
thresh = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, f"{conf_matrix[i, j]}", ha="center", va="center", color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
