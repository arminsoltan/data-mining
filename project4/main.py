import pandas as pd
from sklearn import decomposition, preprocessing, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import numpy as np


def get_train_data():
    data = pd.read_csv("nba_logreg.csv")
    x_train = data.iloc[:, 1: -1]
    y_train = data.iloc[:, -1]
    return x_train, y_train


def get_test_data():
    data = pd.read_csv("test.csv")
    x_test = data.iloc[:, 2: -1]
    y_test = data.iloc[:, -1]
    return x_test, y_test


def pre_processing(x_data):
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    x_data = imp.fit_transform(x_data)
    x_data = preprocessing.StandardScaler().fit_transform(x_data)
    return x_data


def ann(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(max_iter=1000, hidden_layer_sizes=(50, 10), random_state=1)
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    return metrics.accuracy_score(y_test, y_prediction)


def main():
    x_train, y_train = get_train_data()
    x_train = pre_processing(x_train)
    x_test, y_test = get_test_data()
    x_test = pre_processing(x_test)
    print(ann(x_train, y_train, x_test, y_test))


if __name__ == "__main__":
    main()
