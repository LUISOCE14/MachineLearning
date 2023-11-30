import numpy as np

class RedNeuronal:

    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        # Capa oculta
        hidden_inputs = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)

        # Capa de salida con función sigmoide
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs, hidden_outputs

    def backPropagation(self, error, final_inputs, hidden_outputs):

        output_delta = error * self.sigmoid_derivative(final_inputs)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_outputs)

        return output_delta, hidden_delta

    def train(self, X_train, y_train,lr,epochs):
        errors = []
        for epoch in range(epochs):
            # Calculos hacia delante
            final_outputs, hidden_output = self.forward(X_train)

            y_train = y_train.reshape(-1, 1)
            # Calcular error
            error = y_train - final_outputs
            mse = np.mean((error) ** 2)
            errors.append(mse)  # Almacena el MSE en cada época


            # BackPropagation
            output_delta, hidden_delta = self.backPropagation(error, final_outputs, hidden_output)

            # Actualizacion de pesos
            self.weights_hidden_output += hidden_output.T.dot(output_delta) * lr
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * lr
            self.weights_input_hidden += X_train.T.dot(hidden_delta) * lr
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * lr

        return errors

    def predict(self, x):
        # Obtener las salidas de la red
        final_outputs, _ = self.forward(x)

        # Aplicar umbral de 0.5 para obtener salidas binarias
        binary_predictions = (final_outputs >= 0.5).astype(int)

        return binary_predictions
