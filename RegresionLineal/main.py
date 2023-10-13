import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Variables
df = pd.read_csv("salaryDataset.csv")
x = df["YearsExperience"]
y = df["Salary"]
prom_x = np.mean(x)
prom_y = np.mean(y)




# Funcion que calcula la ordenada
def CalcularOrd(x1, y1):
    for xi, yi in zip(x1, y1):
        numerador = np.sum((xi - prom_x) * (yi - prom_y))

    for xi in x1:
        denominador = np.sum((xi - prom_x) ** 2)


    return numerador / denominador


# Calcular los valores de la pendiente y ordenada
m = CalcularOrd(x, y)
b = prom_y - m * prom_x

# Prediccion nueva
valorNew = 10.1
predicionSalario = b + m * valorNew
print("Prediccion del salario cuando tenga ",str(valorNew), "años de experiencia : ",predicionSalario )


# Grafica de ploteo
plt.scatter(x, y, label="Datos Originales")
plt.plot(x, [m * xi + b for xi in x], color="red", linewidth=1, label="Linea de regresion")
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')
plt.title('Regresión Lineal(Salario segun la experiencia)')
plt.legend()
plt.grid(True)

# Agregar el valor predicho en rojo en la gráfica
plt.text(valorNew, predicionSalario, f'Predicción: {predicionSalario:.2f}', color='red', fontsize=9)

# Agregar el punto predicho en el grid
plt.scatter(valorNew, predicionSalario, color='green', marker='o', s=100, label='Predicción')

plt.grid(True)
plt.show()