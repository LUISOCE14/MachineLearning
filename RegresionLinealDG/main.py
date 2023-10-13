import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Variables para la regresión lineal
df = pd.read_csv("salaryDataset.csv")
x = df["YearsExperience"]
y = df["Salary"]

cx = 0.01
iteraciones = 1000

# Función de costo (Mean Squared Error)
def calcular_costo(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Gradiente del Error Cuadrático Medio con respecto a m y b
def calcular_gradiente(X, y, m, b):
    N = len(y)
    gradiente_m = -2/N * np.sum(X * (y - (m * X + b)))
    gradiente_b = -2/N * np.sum(y - (m * X + b))
    return gradiente_m, gradiente_b

# Método del descenso del gradiente para encontrar los coeficientes óptimos
def descenso_del_gradiente(X, y, cx, iteraciones):
    m, b = 0, 0  # Inicializar coeficientes
    for _ in range(iteraciones):
        gradiente_m, gradiente_b = calcular_gradiente(X, y, m, b)
        # Actualizar coeficientes usando el gradiente descendente
        m -= cx * gradiente_m
        b -= cx * gradiente_b
    return m, b


def predecir(x, b, m):
    return m * x + b

# Ejecutar el descenso del gradiente
m_optimo, b_optimo = descenso_del_gradiente(x, y,cx,iteraciones)

# Mostrar los valores óptimos de b y m
print("El valor óptimo de b es:", b_optimo)
print("El valor óptimo de m es:", m_optimo)

#Nueva prediccion
prediccion = predecir(12.0,b_optimo,m_optimo)
print("Prediccion Nueva(Salario de alguien con 6 años de experiencia): ",prediccion)


# Graficar
plt.scatter(x, y, color='blue', label='Datos reales')
plt.plot(x, m_optimo * x + b_optimo, color='red', linewidth=2, label='Línea de regresión')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
# Agregar el valor predicho en rojo en la gráfica
plt.text(6.0, prediccion, f'Predicción: {prediccion:.2f}', color='red', fontsize=9)
# Agregar el punto predicho en el grid
plt.scatter(6.0, prediccion, color='green',  label='Predicción')
plt.legend()
plt.show()





