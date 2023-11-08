import numpy as np
from Perceptron import Perceptron

def accuracy(y_real,y_pred):
    score = np.sum(y_real == y_pred) / len(y_real)
    return score


# Perceptron para OR
training_inputs_or = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
labels_or = np.array([0, 1, 1, 1])

perceptron_or = Perceptron()

perceptron_or.fit(training_inputs_or,labels_or)
y_pred = perceptron_or.predict(training_inputs_or)

xnew2 = np.array([[1,0]])
print(perceptron_or.predict(xnew2))

print("Accuracy for Perceptron OR: ",accuracy(labels_or,y_pred))


#Perceptron para AND
training_inputs_and = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
labels_and = np.array([0, 0, 0, 1])

perceptron_and = Perceptron()

perceptron_and.fit(training_inputs_and,labels_and)
y_pred = perceptron_and.predict(training_inputs_and)
xnew2 = np.array([[1,0]])
print(perceptron_and.predict(xnew2))


print("Accuracy for Perceptron AND: ", accuracy(labels_and,y_pred))






