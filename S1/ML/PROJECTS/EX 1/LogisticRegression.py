from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if self.verbose is True and i % 10000 == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')


    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))


    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

print(X)

model = LogisticRegression(lr=0.1, num_iter=300000)
model.fit(X, y)

preds = model.predict(X, 0.5)
# accuracy
(preds == y).mean()
##visualizing result
theta = model.theta[0]  # Make theta a 1-d array.
x_decision = np.linspace(X[:,0].min(), X[:,0].max(), 50)
y_decision = -(model.intercept_ + theta[0]*x)/theta[1]
#decion boundary
plt.plot(x_decision, y_decision)
#data points
clrs = ['b' if(i==0) else 'r' for i in y]
plt.scatter(X[:,0],X[:,-1],c=clrs)