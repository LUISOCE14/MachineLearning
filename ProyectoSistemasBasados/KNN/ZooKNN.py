from KNN import KNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
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
    num_classes = cm.shape[0]
    classes = np.arange(1, num_classes +1)

    # Ajustar etiquetas y ticks para empezar desde 1
    plt.xticks(np.arange(num_classes) + 0.5, classes)
    plt.yticks(np.arange(num_classes) + 0.5, classes)

    plt.xlabel('Predicciones', fontsize=14)
    plt.ylabel('Etiquetas Verdaderas', fontsize=14)
    plt.title('Matriz de Confusión', fontsize=16)
    plt.show()


#Extraccion de los datos
df = pd.read_csv('zoo.csv')

#Guardar los nombres de los animales
names = df['animal_name']
df = df.drop('animal_name', axis=1)


columns = ['legs']
# Aplicar One-Hot Encoding a las columnas categóricas no binarias y combinar con el DataFrame original
df_encoded = pd.get_dummies(df, columns=columns).astype(int)

# Asigar Carateristicas y Target
X = df_encoded.drop(['class_type'], axis=1).values
y = df_encoded['class_type'].values

# Dividir el data set en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


metrics = Metrics()

k = mejorK(X_train,y_train,X_test,y_test,metrics)

knn2 = KNN(6)
knn2.fit(X_train,y_train)
predictions = knn2.predict(X_test)

#Resultados de metricas
predictions = np.array(predictions)
acurracy = metrics.accuracy(y_test,predictions)
recall = metrics.recall(y_test,predictions)
precision = metrics.precision(y_test,predictions)
f1 = metrics.f1_score(y_test,predictions)
confusionMatrix = metrics.confusion_matrixC(y_test,predictions)



print('Accuracy: ', acurracy)
print('Precision :',precision )
print('Recall: ', recall)
print('F1: ', f1)

plt_confusionMatriz(confusionMatrix)
