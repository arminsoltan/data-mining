from scipy.io import arff
from io import StringIO
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

def get_file_content():
    file = open("PhishingData.arff", "r")
    content = file.read()
    file.close()
    return content


def clean_data(data):
    x_data, y_data = np.zeros((len(data), len(data[0]) - 1)), np.zeros(len(data))

    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            x_data[i][j] = int(data[i][j])
    for i in range(len(data)):
        y_data[i] = int(data[i][-1])
    return x_data, y_data


def get_data():
    content = get_file_content()
    f = StringIO(content)
    data, meta = arff.loadarff(f)
    return clean_data(data)


def algorithm_decision(x_train, y_train, x_test, y_test, algorithm, name):
    classifier = DecisionTreeClassifier(criterion=algorithm)
    classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)
    print("-----------{} Decision tree: -----------------".format(name))
    print("+++++++++++ confusion matrix +++++++++++++++++++")
    print(confusion_matrix(y_test, y_prediction))
    print("+++++++++++ classification result ++++++++++++++")
    print(classification_report(y_test, y_prediction))
    correct = 0
    for i in range(len(y_prediction)):
        if y_test[i] == y_prediction[i]:
            correct += 1
    print("+++++++++++ Accuracy ++++++++++++++++++++")
    print(correct / float(len(y_test)))

def main():
    x_data, y_data = get_data()
    x_test = x_data[:len(x_data) * 3 // 10]
    x_train = x_data[len(x_test):]
    y_test = y_data[:len(y_data) * 3 // 10]
    y_train = y_data[len(y_test):]
    for (algorithm, name) in [("gini","ID3"),("entropy",  "C4.5")]:
        algorithm_decision(x_train, y_train, x_test, y_test, algorithm, name)

if __name__ == '__main__':
    main()