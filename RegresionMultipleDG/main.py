import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Extraccion de datos
df = pd.read_csv("Student_Performance.csv")

df_norm = df.copy()


#Normalizacion de columnas
df_norm['Hours Studied'] = (df['Hours Studied'] - df['Hours Studied'].mean()) / df['Hours Studied'].std()
df_norm['Previous Scores'] = (df['Previous Scores'] - df['Previous Scores'].mean()) / df['Previous Scores'].std()
df_norm['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
df_norm['Sleep Hours'] = (df['Sleep Hours'] - df['Sleep Hours'].mean()) / df['Sleep Hours'].std()
df_norm['Sample Question Papers Practiced'] = (df['Sample Question Papers Practiced'] - df['Sample Question Papers Practiced'].mean()) / df['Sample Question Papers Practiced'].std()


#Asignar Caracteristicas y target
X = df_norm[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
X = np.c_[np.ones((X.shape[0], 1)), X]  # Agregar columna de unos para el término de sesgo (bias)
y = df_norm['Performance Index'].values.reshape(-1, 1)


#Division de los dataFrame en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Inicialización de parámetros y configuración de hiperparámetros
cx = 0.03
iteraciones = 1000
theta_inicial = np.zeros((X.shape[1], 1))

def calcular_Costo(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    error = h - y
    costo = (1 / (2 * m)) * np.sum(error ** 2)
    return costo

def calcular_r2(y_real, y_pred):
    sst = np.sum((y_real - np.mean(y_real)) ** 2)
    ssr = np.sum((y_real - y_pred) ** 2)
    r2 = 1 - (ssr / sst)
    return r2

def calcular_MSE(y_real,y_pre):
    return np.mean((y_pre-y_real)**2)


# Función para realizar el descenso de gradiente
def descenso_gradiente(X, y, theta, cx, iteraciones):
    m = len(y)
    costo_historia = np.zeros(iteraciones)

    for i in range(iteraciones):
        theta = theta - (cx / m) * (X.T.dot(X.dot(theta) - y))
        costo_historia[i] = calcular_Costo(X, y, theta)

    return theta, costo_historia

# Llamada a la función de descenso de gradiente
theta, costo = descenso_gradiente(X_train, y_train, theta_inicial, cx, iteraciones)

print("Coeficientes Descriptivos de cada Variable: \n " ,theta)

# Calcular los valores predichos y_pred usando los coeficientes del modelo
y_pred = X_test.dot(theta)

mse = calcular_MSE(y_test, y_pred)

#Error Cuadrático Medio (MSE)
print("Error Cuadrático Medio (MSE):",mse )

# Calcular el coeficiente de determinación R²
r2 = calcular_r2(y_test,y_pred)
print("Coeficiente de determinación (R²):", r2)

fig, axs = plt.subplots(1, 2,figsize =(12,6))

# Gráfico de Hours Studied vs. Performance Index
fig.suptitle('Regresion Multiple(Variables mas descriptivas)')
axs[0].scatter(X[:, 1], y, color='blue', label='Valores Reales')
axs[0].set_xlabel("Hours Studied")
axs[0].set_ylabel("Performance Index")
axs[0].set_title("Hours Studied vs. Performance Index")

# Dibujar la recta de regresión
x_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
y_values = theta[0] + theta[1] * x_values
axs[0].plot(x_values, y_values, color='green', linewidth=2, label='Recta de Regresión')
plt.legend()


# Gráfico de  Previous score vs. Performance Index

axs[1].scatter(X[:, 2], y, color='blue', label='Valores Reales')
axs[1].set_xlabel("Sleep Hours")
axs[1].set_ylabel("Performance Index")
axs[1].set_title("Sleep Hours vs. Performance Index")

# Dibujar la recta de regresión
x_values = np.linspace(X[:, 2].min(), X[:, 2].max(), 100)
y_values = theta[0] + theta[2] * x_values
axs[1].plot(x_values, y_values, color='green', linewidth=2, label='Recta de Regresión')
plt.legend()
plt.tight_layout()
plt.show()








