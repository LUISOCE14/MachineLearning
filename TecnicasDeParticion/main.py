import pandas as pd
import matplotlib.pyplot as plt
import random

# Definicion de parametros para la particion
particiones = 3
entrenamiento = .70
prueba = .20


df = pd.read_csv('irisbin.csv')

X = df.iloc[:, 0:4]
y = df.iloc[:, 4:7]


# Metodo 1
def particion1(porEntrena, X, Y):
    # Calulamos el numero de filas para entrenamiento y prueba
    numEntrenamiento = int(len(X) * porEntrena)
    numPrueba = len(X) - numEntrenamiento
    # Asignamos las filas y columnas correpondientes
    X_train = X[:numEntrenamiento]
    X_test = X[numEntrenamiento:]
    y_train = Y[:numEntrenamiento]
    y_test = Y[numEntrenamiento:]
    return X_train, X_test, y_train, y_test


# Metodo 2
def particion2(porcentajeEntrena, X, y):
    # Combina las características (X) y el target (y) en un solo DataFrame
    data = pd.concat([X, y], axis=1)

    # Barajar aleatoriamente el conjunto de datos
    data = data.sample(frac=1, random_state=42)

    # Volvemos a dividir el data set
    X1 = data.iloc[:, 0:4]
    y2 = data.iloc[:, 4:7]

    # Calculamos el numero de muestras para cada data set
    numEntrenamiento = int(len(X) * porcentajeEntrena)
    numPrueba = len(X) - numEntrenamiento

    # Asignamos los datos
    X_train = X1[:numEntrenamiento]
    X_test = X1[numEntrenamiento:]
    y_train = y2[:numEntrenamiento]
    y_test = y2[numEntrenamiento:]

    return X_train, X_test, y_train, y_test


# Metodo 3
def particion3(porcentaje, X, y):
    # Combina las características (X) y el target (y) en un solo DataFrame
    data = pd.concat([X, y], axis=1)

    # Barajar aleatoriamente el conjunto de datos
    data = data.sample(frac=1, random_state=42)

    # Calculamos el número de muestras para cada data set
    numEntrenamiento = int(len(data) * porcentaje)

    # Inicializamos listas para almacenar datos de entrenamiento y prueba
    X_entrenamiento = []
    y_entrenamiento = []
    X_prueba = []
    y_prueba = []

    # Dividimos el conjunto de datos en entrenamiento y prueba usando el bucle for
    for i in range(len(data)):
        if i < numEntrenamiento:
            X_entrenamiento.append(data.iloc[i, :-3])  # Primeras 4 columnas para características de entrenamiento
            y_entrenamiento.append(data.iloc[i, -3:])  # Últimas 3 columnas para valores objetivo de entrenamiento
        else:
            X_prueba.append(data.iloc[i, :-3])  # Primeras 4 columnas para características de prueba
            y_prueba.append(data.iloc[i, -3:])  # Últimas 3 columnas para valores objetivo de prueba

    # Convertimos las listas en DataFrames
    X_train = pd.DataFrame(X_entrenamiento)
    y_train = pd.DataFrame(y_entrenamiento)
    X_test = pd.DataFrame(X_prueba)
    y_test = pd.DataFrame(y_prueba)

    return X_train, X_test, y_train, y_test


# Metodo de particion 4
def particion4(porcentaje,x, y,  particion):
    # Combina las características (X) y el target (y) en un solo DataFrame
    data = pd.concat([x, y], axis=1)

    # Barajar aleatoriamente el conjunto de datos
    data = data.sample(frac=1, random_state=42)

    # Calcula los tamaños de los conjuntos de entrenamiento y prueba
    tamano_entrenamiento = int(len(X) * porcentaje)
    tamano_prueba = int(len(X) * prueba)

    # Volvemos a dividir el data set
    X1 = data.iloc[:, 0:4]
    y2 = data.iloc[:, 4:7]

    if particion == 2:
        # Divide los datos en conjuntos de entrenamiento y prueba
        X_train = X1[:tamano_entrenamiento]
        X_test = X1[tamano_entrenamiento:]

        y_train = y2[:tamano_entrenamiento]
        y_test = y2[tamano_entrenamiento:]

        return X_train, X_test, y_train, y_test
    elif particion == 3:
        # Divide los datos en conjuntos de entrenamiento, prueba y validación
        X_train = X1[:tamano_entrenamiento]
        y_train = y2[:tamano_entrenamiento]

        X_test = X1[tamano_entrenamiento:tamano_entrenamiento + tamano_prueba]
        y_test = y2[tamano_entrenamiento:tamano_entrenamiento + tamano_prueba]

        X_validation = X1[tamano_entrenamiento + tamano_prueba:]
        y_validation = y2[tamano_entrenamiento + tamano_prueba:]
        return X_train, X_test, y_train, y_test, X_validation, y_validation


def particionar_kfold(datos, k=5):
    tamano_fold = len(datos) // k
    folds = []
    for i in range(k):
        prueba_inicio = i * tamano_fold
        prueba_fin = prueba_inicio + tamano_fold
        datos_prueba = datos[prueba_inicio:prueba_fin]
        datos_entrenamiento = datos[:prueba_inicio] + datos[prueba_fin:]
        folds.append((datos_entrenamiento, datos_prueba))
    return folds


def Graficar(x_train,y_train,x_test, y_test,x_validation=1,y_validation=1):

    if not isinstance(x_validation, pd.DataFrame):

        fig, axs = plt.subplots(1, 3, figsize=(12, 6))

        # Gráfico de Entrenamiento
        fig.suptitle('Maneras de particionar un data set')
        axs[0].scatter(X_train.iloc[:,0], y_train.iloc[:,0], color='blue', label='Valores Reales')
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        axs[0].set_title(f"Datos de entrenamiento {len(x_train)},{len(y_train)}.")

        # Gráfico de Prueba

        axs[1].scatter(x_test.iloc[:,0], y_test.iloc[:,0], color='blue', label='Valores Reales')
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].set_title(f"Datos de entrenamiento {len(x_test)},{len(y_test)}.")

        plt.show()
    else:
        fig, axs = plt.subplots(2, 2, figsize=(12, 6))

        # Gráfico de Entrenamiento
        fig.suptitle('Maneras de particionar un data set')
        axs[0,0].scatter(X_train.iloc[:, 0], y_train.iloc[:, 0], color='blue', label='Valores Reales')
        axs[0,0].set_xlabel("X")
        axs[0,0].set_ylabel("Y")
        axs[0,0].set_title(f"Datos de entrenamiento {len(x_train)},{len(y_train)}.")

        # Gráfico de Prueba

        axs[0,1].scatter(x_test.iloc[:, 0], y_test.iloc[:, 0], color='blue', label='Valores Reales')
        axs[0,1].set_xlabel("X")
        axs[0,1].set_ylabel("Y")
        axs[0,1].set_title(f"Datos de entrenamiento {len(x_test)},{len(y_test)}.")

        #validation
        axs[1,0].scatter(x_validation.iloc[:, 0], y_validation.iloc[:, 0], color='blue', label='Valores Reales')
        axs[1,0].set_xlabel("X")
        axs[1,0].set_ylabel("Y")
        axs[1,0].set_title(f"Datos de validacion {len(x_validation)},{len(y_validation)}.")
        # Configuración del espacio entre gráficos
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.show()




X_train, X_test, y_train, y_test, X_validation, y_validation = particion4(entrenamiento, X, y, particiones)


Graficar(X_train,y_train,X_test,y_test,x_validation= X_validation,y_validation= y_validation)


