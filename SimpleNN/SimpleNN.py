import numpy as np
from scipy.stats import zscore
from scipy.special import softmax
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


class MyNeuralNetwork:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.w1 = np.random.uniform(-1, 1, (128, 784))
        self.b1 = np.ones((128, 1))
        self.w2 = np.random.uniform(-1, 1, (10, 128))
        self.b2 = np.ones((10, 1))

    def fprop(self, x):
        z1 = np.dot(self.w1, x) + self.b1
        z1 = zscore(z1)
        h1 = sigmoid(z1)
        z2 = np.dot(self.w2, h1) + self.b2
        h2 = softmax(z2)
        return z1, h1, z2, h2

    def update_paramters(self, dW1, db1, dW2, db2):
        self.w1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def one_hot_encoding(self, y):
        y_encoded = np.zeros(10)
        y_encoded[y] = 1
        y_encoded = y_encoded.reshape(10, 1)
        return y_encoded

    def bprop(self, fprop_cache, x, y):
        z1, h1, z2, h2 = fprop_cache
        y = self.one_hot_encoding(y)

        dz2 = h2 - y
        dW2 = np.dot(dz2, h1.T)
        db2 = dz2
        dz1 = np.dot(self.w2.T, (h2 - y)) * sigmoid_der(z1)
        dW1 = np.dot(dz1, x.T)
        db1 = dz1

        self.update_paramters(dW1, db1, dW2, db2)

    def shuffle_X_y(self, X, y):
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        y = y[randomize]
        return X, y

    def fit(self, X, y):
        for e in range(self.epochs):
            X, y = self.shuffle_X_y(X, y)
            for sample, label in zip(X, y):
                sample = sample.reshape(784, 1)
                self.bprop(self.fprop(sample), sample, label)

    def find_y_hat_idx(self, sample):
        sample = sample.reshape(784, 1)
        y_hat = self.fprop(sample)[3]
        y_hat_idx = np.argmax(y_hat)
        return y_hat_idx

    def predict(self, X):
        predictions = [self.find_y_hat_idx(sample) for sample in X]
        return predictions


if __name__ == '__main__':
    # Data - loading and normalizing
    train_x = np.loadtxt(sys.argv[1]).astype(float) / 255
    train_y = np.loadtxt(sys.argv[2]).astype(int)
    test_x = np.loadtxt(sys.argv[3]).astype(float) / 255

    LEARNING_RATE = 0.01
    N_EPOCHS = 36
    my_neural_network = MyNeuralNetwork(LEARNING_RATE, N_EPOCHS)
    my_neural_network.fit(train_x, train_y)
    predictions = my_neural_network.predict(test_x)

    # Writing the result to the output file
    out_f = open(sys.argv[4], "w")
    for i in range(len(test_x)):
        out_f.write(f"{predictions[i]}\n")
    out_f.close()
