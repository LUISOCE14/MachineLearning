import numpy as np


class Metrics:

    
    def confusionMatrix(self,y_true, y_pred):

        true_positives = sum((y_true == 1) & (y_pred == 1))
        false_positives = sum((y_true == 0) & (y_pred == 1))
        true_negatives = sum((y_true == 0) & (y_pred == 0))
        false_negatives = sum((y_true == 1) & (y_pred == 0))

        confusion_matrix_dict = {
            'True Positives': true_positives,
            'False Positives': false_positives,
            'True Negatives': true_negatives,
            'False Negatives': false_negatives
        }

        return confusion_matrix_dict

    def accuracy(self, y_true, y_pred):
        score = np.mean(y_true == y_pred)
        return score

    def precision(self, y_true, y_pred):
        truePositive = sum((y_true == 1) & (y_pred == 1))
        falsePositive = sum((y_true == 0) & (y_pred == 1))

        if truePositive + falsePositive == 0:
            return 0

        precision = truePositive / (truePositive + falsePositive)
        return precision

    def recalll(self, y_true, y_pred):
        truePositive = sum((y_true == 1) & (y_pred == 1))
        falsePositive = sum((y_true == 0) & (y_pred == 1))

        if truePositive + falsePositive == 0:
            return 0

        recall = truePositive / (truePositive + falsePositive)
        return recall

    #Regression

    def f1(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recalll(y_true, y_pred)

        if precision + recall == 0:
            return 0  # Evitar división por cero

        f1 = 2 * ((precision * recall) / (precision + recall))

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

