from sklearn.tree import DecisionTreeClassifier
import csv
from sklearn.metrics import classification_report, confusion_matrix
def get_test_data():
    testdata = csv.reader(open('aps_failure_test_set.csv'))
    testds=[]
    for row in  testdata:
        testds.append(list(row))
    return testds


def eliminate_missing_value(data):
    for i in range(1,len(data[21])):
        pos=0.0
        neg=0.0
        posc=0
        negc=0
        for j in range(21,len(data)):
            if(data[j][0]=='pos'and data[j][i]!='na'):
                pos+=float(data[j][i])
                posc+=1
            elif(data[j][0]=='neg'and data[j][i]!='na'):
                neg+=float(data[j][i])
                negc+=1
        if(posc!=0):
            pos=pos/posc
        if(negc!=0):
            neg=neg/negc
        for j in range(21,len(data)):
            if(data[j][i]=='na' and data[j][0]=='neg'):
                data[j][i]=str(neg)
            if(data[j][i]=='na' and data[j][0]=='pos'):
                data[j][i]=str(pos)
    return(data)


def test_data_divider(testds):
    ds2=[]
    X_test=[]
    for o in range(21,len(testds)):
        x=[]
        for i in range(1,len(testds[21])):
            x.append(testds[o][i])
        X_test.append(x)
    for k in range(len(testds[21])):
        ds3=[]
        for m in range(21, len(testds)):
            ds3.append(testds[m][k])
        ds2.append(ds3)
    Y_test=ds2[0]
    return X_test,Y_test


def get_train_data():
    traindata=csv.reader(open('aps_failure_training_set.csv'))
    trainds=[]
    for row in traindata:
        trainds.append(list(row))
    return trainds


def train_data_driver(trainds):
    ds2=[]
    X_train=[]
    for o in range(21,len(trainds)):
        x2=[]
        for i in range(1,len(trainds[21])):
            x2.append(trainds[o][i])
        X_train.append(x2)
    for k in range(len(trainds[21])):
        ds3=[]
        for m in range(21, len(trainds)):
            ds3.append(trainds[m][k])
        ds2.append(ds3)
    Y_train=ds2[0]
    return X_train,Y_train


def C45_decision_tree(X_train,Y_train,X_test,Y_test):
    classifier = DecisionTreeClassifier(criterion="gini")
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    print("-----------------C4.5 Desision tree:---------------------")
    print("+++++++++++++++++Confution Matrix:+++++++++++++++++++++++")
    print(confusion_matrix(Y_test, y_pred))
    print("+++++++++++++++Classification Result+++++++++++++++++++++")
    print(classification_report(Y_test, y_pred))
    correct = 0
    for i in range(len(Y_test)):
        if (Y_test[i] == y_pred[i]):
            correct += 1
    print("+++++++++++++++++++Accuracy+++++++++++++++++++++++++++++")
    print(correct / float(len(Y_test)) * 100.0)


def IDS3_decision_tree(X_train,Y_train,X_test,Y_test):
    classifier = DecisionTreeClassifier(criterion="entropy")
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    print("-----------------IDS3 Decision tree:---------------------")
    print("+++++++++++++++++Confution Matrix:+++++++++++++++++++++++")
    print(confusion_matrix(Y_test, y_pred))
    print("+++++++++++++++Classification Result+++++++++++++++++++++")
    print(classification_report(Y_test, y_pred))
    correct = 0
    for i in range(len(Y_test)):
        if (Y_test[i] == y_pred[i]):
            correct += 1
    print("+++++++++++++++++++Accuracy+++++++++++++++++++++++++++++")
    print(correct / float(len(Y_test)) * 100.0)



C45_decision_tree(train_data_driver(eliminate_missing_value(get_train_data()))[0],train_data_driver(eliminate_missing_value(get_train_data()))[1],test_data_divider(eliminate_missing_value(get_test_data()))[0],test_data_divider(eliminate_missing_value(get_test_data()))[1])
IDS3_decision_tree(train_data_driver(eliminate_missing_value(get_train_data()))[0],train_data_driver(eliminate_missing_value(get_train_data()))[1],test_data_divider(eliminate_missing_value(get_test_data()))[0],test_data_divider(eliminate_missing_value(get_test_data()))[1])
