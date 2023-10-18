import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Extracción de los datos
df = pd.read_csv("Fish.csv")
df['Species'], uniques = pd.factorize(df['Species'])

# Calcula las correlaciones entre las características
corrMatriz = df.corr("pearson")['Width'].abs().sort_values(ascending=False)
print(corrMatriz)

# Asignación de las características
df['Weight'] = (df['Weight'] - df['Weight'].mean()) / df['Weight'].std()
df['Length3'] = (df['Length3'] - df['Length3'].mean()) / df['Length3'].std()
X = df[["Weight", "Length3"]].values
y = df["Width"].values



# Crear características polinomiales de grado 2
X_poly = np.c_[np.ones(X.shape[0]), X, X**2]

# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# Inicializar parámetros y configuración de descenso de gradiente
theta = np.zeros(X_train.shape[1])
cx = 0.03
iteraciones = 1000

def calcular_Costo(X, y, theta):
    m = len(y)
    error = X.dot(theta) - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

def DescensoGradiente(X, y, theta, tasaAprendizaje, iteracion):
    m = len(y)
    historial_mse = np.zeros(iteracion)

    for i in range(iteracion):
        error = X.dot(theta) - y
        gradient = (1 / m) * X.T.dot(error)
        theta -= tasaAprendizaje * gradient
        historial_mse[i] = calcular_Costo(X, y, theta)

    return theta, historial_mse

def calcular_r2(y_real, y_pred):
    sst = np.sum((y_real - np.mean(y_real)) ** 2)
    ssr = np.sum((y_real - y_pred) ** 2)
    r2 = 1 - (ssr / sst)
    return r2

def calcular_MSE(y_real, y_pred):
    return np.mean((y_pred - y_real) ** 2)

theta, costos = DescensoGradiente(X_train, y_train, theta, cx, iteraciones)

# Calcular los valores predichos y_pred usando los coeficientes del modelo
y_pred = X_test.dot(theta)

# Error Cuadrático Medio (MSE)
mse = calcular_MSE(y_test, y_pred)
print("Error Cuadrático Medio (MSE):", mse)

# Calcular el coeficiente de determinación R²
r2 = calcular_r2(y_test, y_pred)
print("Coeficiente de determinación (R²):", r2)



# Crear un gráfico 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos de datos
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', label='Datos reales')

# Graficar la línea de regresión
x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x_mesh, y_mesh = np.meshgrid(x_values, y_values)
z_mesh = theta[0] + theta[1] * x_mesh + theta[2] * y_mesh + theta[3] * x_mesh**2 + theta[4] * y_mesh**2
ax.plot_surface(x_mesh, y_mesh, z_mesh, color='b', alpha=0.5, label='Línea de regresión')

ax.set_xlabel('Weight')
ax.set_ylabel('Length3')
ax.set_zlabel('Width')
ax.set_title('Regresión Polinómica 3D')

plt.legend()
plt.show()







