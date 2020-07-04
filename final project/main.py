import numpy as np
from sklearn import preprocessing, decomposition
from classifier import gradient_boosting, radius_neighbor, random_forest, ada_boost, knn, mlp, logistic_regression, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from cluster import k_mean_clustering
import pandas as pd


LENGTH_DATA = 1024
DATA_SIZE = 8992
TEST_SIZE = int(0.7 * 8992)


def get_data():
    file = open("qsar_oral_toxicity.csv", "r+")
    all_data = list()
    for line in file:
        data = line.split(';')
        result = data[-1]
        if "negative" in result:
            result = 0
        else:
            result = 1
        data = [int(data[i]) for i in range(LENGTH_DATA)]
        data.append(result)
        all_data.append(data)
    return all_data


def pre_processing(all_data):
    min_max_scalar = preprocessing.MinMaxScaler()
    all_data_minmax = min_max_scalar.fit_transform(all_data)
    return all_data_minmax


def split_data(all_data):
    x_data, y_data = list(), list()
    for data in all_data:
        x_data.append(data[:LENGTH_DATA])
        y_data.append(data[-1])
    x_train = np.array(x_data[:TEST_SIZE])
    y_train = np.array(y_data[:TEST_SIZE])
    x_test = np.array(x_data[TEST_SIZE:])
    y_test = np.array(y_data[TEST_SIZE:])

    return x_train, y_train, x_test, y_test


def reduce_dimensionality(x_data):
    pca = decomposition.PCA(n_components=512)
    return pca.fit_transform(x_data)


def classifying(x_train, y_train, x_test, y_test):
    all_classifier_func = [
        # gradient_boosting,
        radius_neighbor
        # random_forest,
        # ada_boost,
        # knn,
        # mlp,
        # svm,
        # logistic_regression
    ]
    for classifier in all_classifier_func:
        scores = classifier(x_train, y_train, x_test, y_test)
        print("*************************{}********************".format(classifier.__name__))
        for key in scores.keys():
            print("###########################{}########################".format(key))
            print(scores[key])


def clustering(x_train, y_train, x_test, y_test):
    all_clustering_func = [
        k_mean_clustering
    ]
    for cluster in all_clustering_func:
        scores = cluster(x_train, y_train, x_test, y_test)
        print("*************************{}********************".format(cluster.__name__))
        for key in scores.keys():
            print("###########################{}########################".format(key))
            print(scores[key])


def k_fold(x_data, y_data):
    classifiers = [
        gradient_boosting,
        radius_neighbor,
        random_forest,
        ada_boost,
        knn,
        mlp,
        svm,
        logistic_regression
    ]
    accuracy_score = dict()
    for clf in classifiers:
        accuracy_score[clf.__name__] = 0
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(x_data):
        x_train, y_train, x_test, y_test = list(), list(), list(), list()
        for i in train_index:
            x_train.append(x_data[i])
            y_train.append(y_data[i])
        for j in test_index:
            x_test.append(x_data[j])
            y_test.append(y_data[j])
        for clf in classifiers:
            accuracy_score[clf.__name__] += clf(x_train, y_train, x_test, y_test)['accuracy']
    for clf in classifiers:
        print("..............accuracy  {} classifier .............".format(clf.__name__))
        print(accuracy_score[clf.__name__] / 10)


def main():
    all_data = get_data()
    all_data_minmax = pre_processing(all_data)
    x_train, y_train, x_test, y_test = split_data(all_data_minmax)
    x_data = np.concatenate([x_train, x_test])
    x_data = reduce_dimensionality(x_data)
    x_train = x_data[:TEST_SIZE]
    x_test = x_data[TEST_SIZE:]
    classifying(x_train, y_train, x_test, y_test)
    # clustering(x_train, y_train, x_test, y_test)
    # y_data = np.concatenate([y_train, y_test])
    # k_fold(x_data, y_data)


if __name__ == "__main__":
    main()
