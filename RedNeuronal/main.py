from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from RedNeuronal import RedNeuronal
from metrics import Metrics

N = 1000 # muestras
gaussian_quantiles = make_gaussian_quantiles(mean=None,
                        cov=0.1,
                        n_samples=N,
                        n_features=2,
                        n_classes=2,
                        shuffle=True,
                        random_state=None)

X, Y = gaussian_quantiles
Y = Y[:,np.newaxis]

#Division del data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)

#Parametros de la red neuronal
n_entradas = X.shape[1]
n_nOcultas = 4
n_salidas = 1

#Inicializacion de la red
nn = RedNeuronal(n_entradas,n_nOcultas,n_salidas)

#Entrenamiento de la red
nn.train(X_train,y_train,0.01,10000)

# Hacer predicciones en el conjunto de prueba
predictions = nn.predict(X_test)

metrics= Metrics()
#Medicion de la red
ac = metrics.accuracy(y_test,predictions)
precision = metrics.precision(y_test, predictions)
recall = metrics.recalll(y_test,predictions)
f1 = metrics.f1(y_test, predictions)
confusionMatrix = metrics.confusionMatrix(y_test,predictions)

print("Accuracy:", ac)
print("Precision: ", precision)
print("Recall:", recall)
print("F1: ", f1)

#Ploteo de prediciones y valores reales
plt.scatter(X_test[:, 0], y_test, label='Real')
plt.scatter(X_test[:, 0], predictions,  label='Predicción')
plt.show()











