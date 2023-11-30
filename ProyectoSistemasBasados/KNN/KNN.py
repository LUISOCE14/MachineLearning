import numpy as np

class KNN:
    def __init__(self,k=2):
        self.k = k

    def fit(self,X,y):
        self.X_train  = X
        self.y_train  = y

    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]
        most_common = np.bincount(k_neighbor_labels).argmax()
        return most_common
