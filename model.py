import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=2000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0

        for _ in range(self.epochs):
            linear = np.dot(X, self.W) + self.b
            y_pred = sigmoid(linear)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if i >= 0.5 else 0 for i in probs]
