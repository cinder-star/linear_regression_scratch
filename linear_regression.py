import numpy as np
import pandas as pd
import random
import json
import matplotlib.pyplot as plt


class Model(object):
    n = 0
    theta = None

    def __init__(self):
        super().__init__()

    def _evaluate(self, X, theta, y):
        x = np.dot(X, theta) - y
        return x

    def train(self, data, epoch=10, verbose=True, batch_size=None, learning_rate=1.0):
        for row in data:
            row.append(row[-1])
            row[-2] = 1
        data = np.array(data, dtype=np.float32)
        n = data.shape[1]
        if not self.theta or self.theta.shape[0] != n-1:
            self.theta = np.random.rand(n - 1, 1)
        if not batch_size:
            batch_size = 10
        steps = data.shape[0] // batch_size + 1
        epoches = [int(x) for x in range(epoch)]
        errors = []
        for i in range(epoch):
            error = 0.0
            for j in range(steps):
                raw_data = []
                y = []
                for k in range(
                    j * batch_size, min(data.shape[0], (j + 1) * batch_size)
                ):
                    raw_data.append(data[k][:-1])
                    y.append([data[k][-1]])
                raw_data = np.array(raw_data, dtype=np.float32)
                y = np.array(y, dtype=np.float32)
                if len(raw_data) > 0 and len(y) > 0:
                    x = self._evaluate(raw_data, self.theta, y)
                    error = error + np.dot(np.transpose(x), x)[0][0]
                    transpose = np.transpose(raw_data)
                    self.theta = self.theta - np.dot(transpose, x) * 2 * learning_rate
            errors.append(error)
            if verbose:
                print("epoch: {} error: {}".format(i, error))
        return {"epoches": epoches, "errors": errors}

    def predict(self, data):
        data.append(1.0)
        raw_data = np.array([data], dtype=np.float32)
        return np.dot(raw_data, self.theta)[0][0]

    def load(self, filepath):
        file = open(filepath, "r")
        values = json.load(file)
        self.theta = []
        for key, value in values.items():
            self.theta.append([value])
        self.theta = np.array(self.theta, dtype=np.float32)

    def save(self, filepath):
        values = {}
        file = open(filepath, "w")
        i = 0
        print(self.theta)
        for value in self.theta:
            values[str(i)] = value[0]
            i = i+1
        json.dump(values, file)

if __name__ == ("__main__"):
    regression = Model()
    data = []
    x_cut = []
    y_cut = []
    for i in range(400):
        x = random.random()*20+1
        z = 2*x + 5.0 + random.random()*7.0
        x_cut.append(x)
        y_cut.append(z)
        data.append([x, z])
    # result = regression.train(data=data, batch_size=2, epoch=400, learning_rate=0.0002, verbose=False)
    # regression.save("regression.json")
    regression.load("regression.json")
    x = []
    y = []
    for i in range(20):
        x.append(i+1)
        y.append(regression.predict([i+1]))
    plt.plot(x, y, "r-")
    categories = []
    for i in range(400):
        categories.append(random.randrange(0,3))
    colormap = np.array(["r", "g", "b"])
    plt.scatter(x_cut, y_cut, c=colormap[categories], s=2)
    plt.grid()
    plt.show()