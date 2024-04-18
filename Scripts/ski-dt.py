import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import sys

def decision_tree_classifier(X_train, X_test, y_train, y_test, random_state=0, max_depth=6, criterion='gini'):
    """
    Train and test a decision tree classifier.

    Parameters:
    X_train (array-like): Training data, features.
    X_test (array-like): Test data, features.
    y_train (array-like): Training data, labels.
    y_test (array-like): Test data, labels.
    max_depth (int): The maximum depth of the tree. Default is 6.
    criterion (str): The function to measure the quality of a split. Default is 'gini'.

    Returns:
    float: Accuracy of the model on the test set.
    """

    classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    return accuracy_score(y_test, y_pred)





dataset = sys.argv[1]


file_name = './Data/%s_%s.csv' % (dataset, "train")
train_data = pd.read_csv(file_name, header=None)
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_train = X_train.astype("int")
y_train = y_train.astype("int")

file_name = './Data/%s_%s.csv' % (dataset, "test")
test_data = pd.read_csv(file_name, header=None)
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
X_test = X_test.astype("int")
y_test = y_test.astype("int")

# 使用函数处理 Iris 数据集
accuracy = []
for j in range(5):
    accuracy.append(decision_tree_classifier(X_train, X_test, y_train, y_test, random_state=random.randint(0, 20)))
print(f"scikit-learn accuracy on {dataset}: {sum(accuracy)/5}")
