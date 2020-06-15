from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn import metrics


def knn(x_train, y_train, x_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_prediction = neigh.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
    return metrics.accuracy_score(y_test, y_prediction), confusion_matrix


def random_forest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(max_depth=300)
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
    return metrics.accuracy_score(y_test, y_prediction), confusion_matrix


def gradient_boosting(x_train, y_train, x_test, y_test):
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
    return metrics.accuracy_score(y_test, y_prediction), confusion_matrix


def ada_boost(x_train, y_train, x_test, y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
    return metrics.accuracy_score(y_test, y_prediction), confusion_matrix


def mlp(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
    accuracy_score = metrics.accuracy_score(y_test, y_prediction)
    precision_score = metrics.precision_score(y_test, y_prediction)
    recall_score = metrics.recall_score(y_test, y_prediction)
    f_measure = metrics.f1_score(y_test, y_prediction)
    scores = {
        'accuracy': confusion_matrix,
        'confusion': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f_measure': f_measure
    }
    return scores


def radius_neighbor(x_train, y_train, x_test, y_test):
    clf = RadiusNeighborsClassifier(radius=30)
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
    return metrics.accuracy_score(y_test, y_prediction), confusion_matrix



