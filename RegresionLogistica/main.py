import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from RegresionLogistica import LogisticRegression

# Extraccion de los datos
df = pd.read_csv('Employee.csv')

# label encoding
df['Education'] = df['Education'].astype('category').cat.codes
df['City'] = df['City'].astype('category').cat.codes
df['Gender'] = df['Gender'].astype('category').cat.codes
df['EverBenched'] = df['EverBenched'].astype('category').cat.codes

# Aiginamos las carateristicas y el target
X = df.drop("LeaveOrNot", axis="columns")
y = df["LeaveOrNot"]

# estandarizacion de los datos
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))

# Dividimos el data set en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def MSE(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def r2_score(y_true, y_pred):
    ss_total = ((y_true - y_true.mean()) ** 2).sum()
    ss_residual = ((y_true - y_pred) ** 2).sum()
    r2 = 1 - (ss_residual / ss_total)
    return r2


clf = LogisticRegression(lr=0.03)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("MSE: ", MSE(y_test,y_pred))
print("R2: ",r2_score(y_test,y_pred))

# Suponiendo que X_test y y_pred son DataFrames de pandas
x_values = X_test.iloc[:, 0]
y_true_values = y_test
y_pred_values = y_pred

plt.scatter(x_values, y_true_values, color="red", label="Datos reales", marker="o")
plt.scatter(x_values, y_pred_values, color="blue", label="Datos reales", marker="X")

plt.xlabel("Primer Atributo")
plt.ylabel("Variable de Interés")
plt.legend()
plt.show()





