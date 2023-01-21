import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def mode_robustness(x_test, y_test, model, std_coef, number_prediction=200):
     result = []
     for std in std_coef:
        print("std: {}".format(std))
        x_class = x_test
        mean = 0
        sigma = np.std(x_class) * std
        gaussian = np.random.normal(mean, sigma, x_class.shape)
        x_class = x_class + gaussian

        mc_predictions = []

        for _ in range(number_prediction):
            y_p_class = model.predict(x_class)
            mc_predictions.append(y_p_class)

        mc_ensemble_pred = np.array(mc_predictions).mean(axis=0).argmax(axis=1)
        ensemble_acc = accuracy_score(y_test.argmax(axis=1), mc_ensemble_pred)
        ensemble_precision = precision_score(y_test.argmax(axis=1), mc_ensemble_pred, average='weighted')
        ensemble_recall = recall_score(y_test.argmax(axis=1), mc_ensemble_pred, average='weighted')
        ensemble_F1 = (2 * ensemble_precision * ensemble_recall) / (ensemble_precision + ensemble_recall)

        result = [ensemble_acc * 100, ensemble_precision * 100, ensemble_recall * 100, ensemble_F1 * 100]
        print("[{:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}]\n".format(*result))


def evaluation(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    F1 = (2 * precision * recall) / (precision + recall)
    return acc, precision, recall, F1
