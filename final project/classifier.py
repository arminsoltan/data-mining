from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn import svm as svm_clf
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def knn(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    return compute_validation(x_train, y_train, x_test, y_test, clf)


def random_forest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(max_depth=300)
    return compute_validation(x_train, y_train, x_test, y_test, clf)


def gradient_boosting(x_train, y_train, x_test, y_test):
    clf = GradientBoostingClassifier(random_state=0)
    return compute_validation(x_train, y_train, x_test, y_test, clf)


def ada_boost(x_train, y_train, x_test, y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    return compute_validation(x_train, y_train, x_test, y_test, clf)


def mlp(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(40, 10))
    return compute_validation(x_train, y_train, x_test, y_test, clf)


def radius_neighbor(x_train, y_train, x_test, y_test):
    clf = RadiusNeighborsClassifier(radius=200)
    return compute_validation(x_train, y_train, x_test, y_test, clf)


def svm(x_train, y_train, x_test, y_test):
    clf = svm_clf.SVC(gamma='scale')
    return compute_validation(x_train, y_train, x_test, y_test, clf)


def logistic_regression(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(random_state=0)
    return compute_validation(x_train, y_train, x_test, y_test, clf)


def compute_validation(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
    accuracy_score = metrics.accuracy_score(y_test, y_prediction)
    precision_score = metrics.precision_score(y_test, y_prediction)
    recall_score = metrics.recall_score(y_test, y_prediction)
    f_measure = metrics.f1_score(y_test, y_prediction)
    scores = {
        'accuracy': accuracy_score,
        'confusion': confusion_matrix,
        'precision': precision_score,
        'recall': recall_score,
        'f_measure': f_measure
    }
    return scores
