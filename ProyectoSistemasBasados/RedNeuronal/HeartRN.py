from RedNeuronal import RedNeuronal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from metrics import Metrics

def plt_error(error):
    # Graficar el progreso del error
    plt.plot(range(10000), errors, label=str(min(errors)))
    plt.title('Progreso del Error durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Error Medio Absoluto')
    plt.legend()
    plt.show()

def  plt_confusionMatriz(cm):
    # Crear un mapa de calor usando seaborn
    plt.figure(figsize=(8, 6))
    sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5, cbar=False)

    # Configuraciones adicionales para mejorar la visualización
    plt.xlabel('Predicciones', fontsize=14)
    plt.ylabel('Etiquetas Verdaderas', fontsize=14)
    plt.title('Matriz de Confusión', fontsize=16)
    plt.show()


df = pd.read_csv('heart.csv')


columns = ['cp','slp','thall']
# Aplicar One-Hot Encoding a las columnas categóricas no binarias y combinar con el DataFrame original
df_encoded = pd.get_dummies(df, columns=columns).astype(int)

# Asigar Carateristicas y Target
X = df.drop(['output','sex'], axis=1)
y = df['output'].values
y = y.reshape(-1, 1)

# Escalar datos para que esten en el rango de los otros
scaler = MinMaxScaler()

# Aplicar la normalización a los datos
X = scaler.fit_transform(X)

input_size = X.shape[1]
hidden_size = 200
output_size = 1


# Dividir el data set en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Red neuronal
rn = RedNeuronal(input_size,hidden_size,output_size)
errors = rn.train(X_train, y_train, lr=0.003, epochs=10000)
plt_error(errors)
y_pred = rn.predict(X_test)

#Resultados de metricas
metrics = Metrics()
predictions = np.array(y_pred)
print(y_pred.shape,y_test.shape)
acurracy = metrics.accuracy(y_test,predictions)
recall = metrics.recall(y_test,predictions)
precision = metrics.precision(y_test,predictions)
f1 = metrics.f1(y_test,predictions)
confusionMatrix = metrics.confusionMatrix(y_test,predictions)
confusionMatrix = np.squeeze(confusionMatrix)

print('Accuracy: ', acurracy)
print('Precision :',precision )
print('Recall: ', recall)
print('F1: ', f1)

plt_confusionMatriz(confusionMatrix)