import numpy as np


class Metrics:

    def confusion_matrixC(self, y_true, y_pred):
        num_classes = len(np.unique(y_true))+1
        # Inicializar la matriz de confusión con ceros
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Llenar la matriz de confusión
        for true_label, pred_label in zip(y_true, y_pred):
            conf_matrix[true_label - 1, pred_label - 1] += 1

        return conf_matrix

    def confusionMatrix(self,y_true, y_pred):

        true_positives = sum((y_true == 1) & (y_pred == 1))
        false_positives = sum((y_true == 0) & (y_pred == 1))
        true_negatives = sum((y_true == 0) & (y_pred == 0))
        false_negatives = sum((y_true == 1) & (y_pred == 0))

        confusion_matrix = [[true_positives,false_negatives],[false_positives,true_negatives]]
        confusion_matrix = np.array(confusion_matrix)

        return confusion_matrix

    def accuracy(self, y_true, y_pred):
        score = np.mean(y_true == y_pred)
        return score


    def precision(self, y_true, y_pred):
        true_positives = sum((y_true == 1) & (y_pred == 1))
        false_positives = sum((y_true == 0) & (y_pred == 1))
        true_negatives = sum((y_true == 0) & (y_pred == 0))
        false_negatives = sum((y_true == 1) & (y_pred == 0))

        if true_positives + false_positives == 0:
            return 0

        precision = (true_positives + true_negatives) / (true_positives + false_positives +
                                                         true_negatives + false_negatives)
        return precision

    def recall(self,y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        if true_positives + false_negatives == 0:
            return 0

        recall = true_positives / (true_positives + false_negatives)
        return recall

    def f1_score(self,y_true, y_pred):
        precision_val = self.precision(y_true, y_pred)
        recall_val = self.recall(y_true, y_pred)

        if precision_val + recall_val == 0:
            return 0

        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        return f1

    def mse(self, y_true,y_pred):
        n =len(y_true)
        mse = sum((y_pred - y_true)**2) / n
        return mse

    def mae(self, y_true, y_pred):
        n = len(y_true)
        mae = sum(y_pred - y_true) / n
        return mae

    def rmse(self, y_true, y_pred):
        rmse = np.sqrt(self.mse(y_true, y_pred))
        return rmse

