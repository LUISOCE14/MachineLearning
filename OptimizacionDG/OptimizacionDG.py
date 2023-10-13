import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la función F(x1, x2)
def funcionObj(x1, x2):
    return 10 - np.exp(-(x1**2 + 3*x2**2))

# Calcular el gradiente de la función F(x1, x2)
def gradiente(x1, x2):
    df_dx1 = 2 * x1 * np.exp(-(x1**2 + 3*x2**2))
    df_dx2 = 6 * x2 * np.exp(-(x1**2 + 3*x2**2))
    return np.array([df_dx1, df_dx2])

# Método del descenso del gradiente para optimizar la función y registrar los saltos
def gradient_descent(learning_rate, num_iterations):
    x = np.random.uniform(-1, 1, size=(2,)) # Inicializar x1 y x2 en un valor aleatorio entre -1 y 1
    puntosVisi = [x.copy()]  # Almacenar los puntos para la visualización
    for _ in range(num_iterations):

        grad = gradiente(x[0], x[1])  # Calcular el gradiente en el punto actual

        x = x - learning_rate * grad  # Actualizar los valores de x usando el descenso del gradiente

        puntosVisi.append(x.copy())  # Registrar el punto para la visualización

    return np.array(puntosVisi)


def graficar(visitados):
    # Crear una malla de puntos para la visualización en 3D
    x1 = np.linspace(-1, 1, 400)
    x2 = np.linspace(-1, 1, 400)
    X1, X2 = np.meshgrid(x1, x2)
    # optimizacion con funcion objivo
    Z = funcionObj(X1, X2)

    # Graficar la función en 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis')  # Superficie tridimensional de la función
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('F(x1, x2)')
    ax.set_title('Optimizacion del la funcion')

    # Graficar los saltos durante el descenso del gradiente
    ax.plot(visitados[:, 0], visitados[:, 1], funcionObj(visitados[:, 0], visitados[:, 1]), marker='o', color='red',
            markersize=5, label='Descenso del gradiente')
    ax.legend()

    plt.show()



# Configuración de parámetros
lr = 0.1
num_iter = 100

# Optimizar la función utilizando el descenso del gradiente y registrar los saltos
visitados = gradient_descent(lr, num_iter)

#Mostrar Grafica
graficar(visitados)