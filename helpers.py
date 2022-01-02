import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.neural_network import MLPClassifier


def analyse(y_test, y_pred, show=False):

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = accuracy_score(y_test, y_pred)

    if(show):
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()
        print("accuracy = {:.4f}, sensitivity = {:.4f}, specificity = {:.4f}".format(
            accuracy, sensitivity, specificity))

    return accuracy, sensitivity, specificity


def computeDecitionTree(X_train, X_test, y_train, **params):
    decisionTree = DecisionTreeClassifier(**params, random_state=0)
    decisionTree.fit(X_train, y_train)
    y_pred = decisionTree.predict(X_test)

    dot_data = tree.export_graphviz(decisionTree, out_file=None, max_depth=3,
                                    feature_names=X_train.columns, class_names=list(
                                        map(str, sorted(y_train.unique()))),
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)

    return y_pred, graph


def computeMLP(X_train, X_test, y_train, **params):
    mlp = MLPClassifier(**params, max_iter=400, random_state=0)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    return y_pred
