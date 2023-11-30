from KNN import KNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from metrics import Metrics

def mejorK(X_train, y_train, X_test, y_test,mt):
    scoreList = []
    presicion = []
    for i in range(1, 20):
        knn = KNN(k=i)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        scoreList.append(mt.accuracy(y_test, predictions))
        presicion.append(mt.precision(y_test, np.array(predictions)))

    # Plot de los resultados
    plt.plot(range(1, 20), scoreList, label="Accuracy", marker='o', color="red", linewidth=3)
    plt.plot(range(1, 20), presicion, color='blue', label="Precision")
    plt.xticks(np.arange(1, 20, 1))
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    indice_maximo, maximo = max(enumerate(scoreList), key=lambda x: x[1])

    return indice_maximo+1


def  plt_confusionMatriz(cm):
    # Crear un mapa de calor usando seaborn
    plt.figure(figsize=(8, 6))
    sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5, cbar=False)

    # Configuraciones adicionales para mejorar la visualización
    plt.xlabel('Predicciones', fontsize=14)
    plt.ylabel('Etiquetas Verdaderas', fontsize=14)
    plt.title('Matriz de Confusión', fontsize=16)
    plt.show()


df = pd.read_csv('/EmployeKNN.py')


# Obtener las columnas categóricas (excluyendo aquellas que ya son binarias)
non_binary_categorical_columns = df.select_dtypes(include=['object']).columns
# Aplicar One-Hot Encoding a las columnas categóricas no binarias y combinar con el DataFrame original
df_encoded = pd.get_dummies(df, columns=non_binary_categorical_columns).astype(int)
df = df_encoded

# Asigar Carateristicas y Target
X = df.drop('LeaveOrNot', axis=1).values
y = df['LeaveOrNot'].values



# Escalar datos para que esten en el rango de los otros
scaler = MinMaxScaler()
# Aplicar la normalización a los datos
X = scaler.fit_transform(X)

# Dividir el data set en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

metrics = Metrics()

#Encontrar el mejor numero de vecinos
k = mejorK(X_train, y_train, X_test, y_test, metrics)


knn2 = KNN(k)
knn2.fit(X_train,y_train)
Y_pred = knn2.predict(X_test)

#Resultados de metricas
predictions = np.array(Y_pred)
acurracy = metrics.accuracy(y_test,predictions)
recall = metrics.recall(y_test,predictions)
precision = metrics.precision(y_test,predictions)
f1 = metrics.f1_score(y_test,predictions)
confusionMatrix = metrics.confusionMatrix(y_test,predictions)


print('Accuracy: ', acurracy)
print('Precision :',precision )
print('Recall: ', recall)
print('F1: ', f1)

plt_confusionMatriz(confusionMatrix)