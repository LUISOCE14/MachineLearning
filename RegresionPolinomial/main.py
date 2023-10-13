import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("Regresion.csv")
# Carateristicas
X = df["araba_fiyat"]
#Target
y = df["araba_max_hiz"]

print(X)

#Grado del polinomio
grado = 3

#Variable para guardar el error cuadratico
mse = 0.0
SSE = 0.0
SST = 0.0



#leave-one out
for i in range(0,10):

    #Separacion de dataframe en entrenamiento y prueba
    X_train = np.delete(X, i)
    Y_train = np.delete(y, i)

    x_test = X[i]
    y_test = y[i]

    # Calculando la regresión polinomial
    coeficiente = np.polyfit(X_train,Y_train,grado)
    polinomio = np.poly1d(coeficiente)

    #Calcular el error cuadratico
    mse += (polinomio(x_test) - y_test) ** 2

    # Acumulando sumas para el cálculo de R^2
    SSE += (polinomio(x_test) - y_test) ** 2
    SST += (y_test - np.mean(Y_train)) ** 2


mse /= len(X)
r_Squared = 1 - (SSE / SST)

# Imprimir el error cuadrático medio y R^2
print("Error Cuadrático Medio (MSE) promedio: ", mse)
print("Coeficiente de Determinación (R^2): ", r_Squared)

# Calculando la regresión polinómica utilizando todos los datos
coefficients = np.polyfit(X, y, grado)
polynomial = np.poly1d(coefficients)

new_x = 2500

#Predecir nuevos puntos
new_Y = polynomial(new_x)

# Plotear los puntos y la línea de regresión polinomial con todos los datos
plt.scatter(X, y, color='blue', label='Datos')
X_plot = np.linspace(min(X), max(X), 100)
plt.plot(X_plot, polynomial(X_plot), color='red', label='Regresión Polinomial')
plt.scatter(new_x,new_Y,color="green")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión Polinomial de Grado ' + str(grado) )
plt.legend()
plt.show()





