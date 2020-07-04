from sklearn.cluster import KMeans
from sklearn import metrics


def k_mean_clustering(x_train, y_train, x_test, y_test):
    k_means = KMeans(n_clusters=2, random_state=0)
    k_means.fit(x_train, y_train)
    y_predict = k_means.predict(x_test)
    return _compute_score(y_test, y_predict)


def _compute_score(y_test, y_predict):
    confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
    accuracy_score = metrics.accuracy_score(y_test, y_predict)
    precision_score = metrics.precision_score(y_test, y_predict)
    recall_score = metrics.recall_score(y_test, y_predict)
    f_measure = metrics.f1_score(y_test, y_predict)
    scores = {
        'accuracy': accuracy_score,
        'confusion': confusion_matrix,
        'precision': precision_score,
        'recall': recall_score,
        'f_measure': f_measure
    }
    return scores

