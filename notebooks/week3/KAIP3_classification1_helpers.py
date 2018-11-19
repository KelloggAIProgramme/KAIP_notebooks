import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#import graphviz
from sklearn import tree

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_models_iris(models, X, y):
    # import some data to play with
    X_sk = X.values[:,:2]
    y_sk = list(y.values)
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    for i in range(len(y_sk)):
        if y_sk[i] == 'Iris-setosa':
            y_sk[i] = 0
        elif y_sk[i] == 'Iris-versicolor':
            y_sk[i] = 1
        elif y_sk[i] == 'Iris-virginica':
            y_sk[i] = 2
    models = (model.fit(X_sk, y_sk),)

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors

    models = (clf.fit(X_sk, y_sk) for clf in models)

    # title for the plots
    titles = ('Nearest neighbour classification',
              'Logistic regression', 'Linear SVM', 'Decision Tree')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X_sk[:, 0], X_sk[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    iter_sub = iter(sub.flatten())

    #ax_first = next(iter_sub)
    #ax_first.scatter(X0, X1, c=y_sk, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #ax_first.set_xlim(xx.min(), xx.max())
    #ax_first.set_ylim(yy.min(), yy.max())
    #ax_first.set_xlabel('Sepal length', fontsize = 14)
    #ax_first.set_ylabel('Sepal width', fontsize = 14)
    #ax_first.set_xticks(())
    #ax_first.set_yticks(())
    #ax_first.set_title('Dataset')

    for clf, title, ax in zip(models, titles, iter_sub):

        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y_sk, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length', fontsize = 14)
        ax.set_ylabel('Sepal width', fontsize = 14)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)


    plt.show()

def plot_model_iris(model, X, y):
    # import some data to play with
#    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    #X_sk = iris.data[:, :2]
    #y_sk = iris.target
    X_sk = X.values[:,:2]
    y_sk = list(y.values)
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    for i in range(len(y_sk)):
        if y_sk[i] == 'Iris-setosa':
            y_sk[i] = 0
        elif y_sk[i] == 'Iris-versicolor':
            y_sk[i] = 1
        elif y_sk[i] == 'Iris-virginica':
            y_sk[i] = 2
    models = (model.fit(X_sk, y_sk),)

    # title for the plots
    titles = ('Model',)

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X_sk[:, 0], X_sk[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    iter_sub = iter(sub.flatten())

    ax_first = next(iter_sub)
    ax_first.scatter(X0, X1, c=y_sk, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax_first.set_xlim(xx.min(), xx.max())
    ax_first.set_ylim(yy.min(), yy.max())
    ax_first.set_xlabel('Sepal length', fontsize = 14)
    ax_first.set_ylabel('Sepal width', fontsize = 14)
    ax_first.set_xticks(())
    ax_first.set_yticks(())
    ax_first.set_title('Dataset')

    for clf, title, ax in zip(models, titles, iter_sub):

        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y_sk, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length', fontsize = 14)
        ax.set_ylabel('Sepal width', fontsize = 14)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)


    plt.show()

'''def plot_tree(clf_tree):

    # import some data to play with
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    clf_tree = clf_tree.fit(X,y)
    dot_data = tree.export_graphviz(clf_tree, out_file=None,
                             feature_names=iris.feature_names[:2],
                             class_names=iris.target_names,
                             filled=True, rounded=True,
                             special_characters=True)
    graph = graphviz.Source(dot_data)
    return graph
'''
