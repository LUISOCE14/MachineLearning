import numpy as np


class Perceptron:
    def __init__(self, lr=0.01, epochs=1000, tol=1e-4):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.weights = None
        self.bias = None

    def activation(self, input):
        res = 1 / (1 + np.exp(-input))
        return np.where(res <= 0.5, 0, 1)

    def fit(self, x, y):
        n_row, n_columns = x.shape
        # Inicializacion de parametros
        self.weights = np.zeros(n_columns)
        self.bias = 0
        # Copia de los pesos para comparación
        prev_weights = np.copy(self.weights)
        for i in range(self.epochs):
            for inputs, label in zip(x, y):
                y_pred = self.predict(inputs)
                self.update_Weights(inputs, label, y_pred)
                # Verificar el criterio de paro
            if np.all(np.abs(self.weights - prev_weights) < self.tol):
                break
            prev_weights = np.copy(self.weights)

    def update_Weights(self, inputs, y_real, y_pred):
        error = y_real - y_pred
        weigthCorrection = self.lr * error
        self.weights = self.weights + weigthCorrection * inputs
        self.bias = self.bias + weigthCorrection

    def predict(self, x):
        output = np.dot(x, self.weights) + self.bias
        y_pred = self.activation(output)
        return y_pred
