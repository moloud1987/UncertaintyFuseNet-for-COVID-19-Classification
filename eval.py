import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def mode_robustness(x_test, y_test, model, std_coef):
    result = []
    for std in std_coef:

        y_class = y_test
        x_class = x_test
        mean = np.mean(x_class) * 0
        sigma = np.std(x_class) * std
        gaussian = np.random.normal(mean, sigma, x_class.shape)
        x_class = x_class + gaussian
        y_p_class = model.predict(x_class)
        acc = accuracy_score(np.argmax(y_class, axis=1), np.argmax(y_p_class, axis=1))
        precision = precision_score(np.argmax(y_class, axis=1), np.argmax(y_p_class, axis=1), average='weighted')
        recall = recall_score(np.argmax(y_class, axis=1), np.argmax(y_p_class, axis=1), average='weighted')
        F1 = (2 * precision * recall) / (precision + recall)
        result.append([std, acc, precision, recall, F1])

    return result


def evaluation(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    F1 = (2 * precision * recall) / (precision + recall)
    return acc, precision, recall, F1