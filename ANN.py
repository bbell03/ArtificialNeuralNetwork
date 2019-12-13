import numpy as np
import pandas as pd


def encodersetosa(Class):
    if Class == "Iris-setosa":
        return 1
    else:
        return 0


def encoderversicolor(Class):
    if Class == "Iris-versicolor":
        return 1
    else:
        return 0


def encodervirginica(Class):
    if Class == "Iris-virginica":
        return 1
    else:
        return 0


def sig(x):
    return (1 / (1 + np.exp(-x)))


def derisig(y):
    return y * (1 - y)


class NerualNetwork:
    Input_Size = 0
    Hidden_Size = 0
    Output_Size = 0
    w0 = None
    w1 = None
    b0 = None
    b1 = None
    lr = None

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate):
        self.Input_Size = input_layer_size
        self.Hidden_Size = hidden_layer_size
        self.Output_Size = output_layer_size
        self.w0 = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
        self.w1 = 2 * np.random.random((hidden_layer_size, output_layer_size)) - 1
        self.b0 = 1
        self.b1 = 1
        self.lr = learning_rate
        pass

    def train(self, X, y, iter_turns):
        for iter in range(iter_turns):
            for i in range(len(X)):
                # forward propagation
                self.l0 = X[i]
                y_one = y[i]
                self.l1 = sig(np.dot(self.l0, self.w0) + self.b0)
                self.l2 = sig(np.dot(self.l1, self.w1) + self.b1)
                Delta2 = derisig(self.l2) * (y_one - self.l2)

                # backword propagation
                Delta1 = derisig(self.l1) * np.dot(self.w1, Delta2)
                self.w1 += np.dot(self.l1.reshape(self.Hidden_Size, 1), Delta2.reshape(1, self.Output_Size)) * self.lr
                self.b1 += Delta2 * 1 * self.lr
                self.w0 += np.dot(self.l0.reshape(self.Input_Size, 1), Delta1.reshape(1, self.Hidden_Size)) * self.lr
                self.b0 += Delta1 * 1 * self.lr

        print("Finish Training, IterTurn = " + str(iter_turns))

    def predict(self, X):
        l0 = X
        l1 = sig(np.dot(l0, self.w0) + self.b0)
        l2 = sig(np.dot(l1, self.w1) + self.b1)
        Result = l2.tolist()
        result = []
        for i in Result:
            if (i[0] >= i[1]) & (i[0] >= i[2]):
                result.append([1, 0, 0])
            elif (i[1] >= i[0]) & (i[1] >= i[2]):
                result.append([0, 1, 0])
            elif (i[2] >= i[0]) & (i[2] >= i[1]):
                result.append([0, 0, 1])
        return np.array(result)

    def predict_and_calculate_accuracy(self, X, y):
        result = self.predict(X).tolist()
        count = 0
        for i in range(len(result)):
            if result[i] != y.tolist()[i]:
                count += 1
        return 1 - (count / len(result))

    def predict_one(self, X):
        result = ""
        l0 = [X]
        l1 = sig(np.dot(l0, self.w0) + self.b0)
        l2 = sig(np.dot(l1, self.w1) + self.b1)
        Result = l2.tolist()
        for i in Result:
            if (i[0] >= i[1]) & (i[0] >= i[2]):
                result = "Iris-setosa"
            elif (i[1] >= i[0]) & (i[1] >= i[2]):
                result = "Iris-versicolor"
            elif (i[2] >= i[0]) & (i[2] >= i[1]):
                result = "Iris-virginica"
        return result


if __name__ == "__main__":
    df = pd.read_csv("ANN - Iris data.txt", header=None)
    df = df.rename(columns={0: "Sepal-Length", 1: "Sepal-Width", 2: "Petal-Length", 3: "Petal-Width", 4: "Class"})

    df["Iris-setosa"] = df["Class"].apply(encodersetosa)
    df["Iris-versicolor"] = df["Class"].apply(encoderversicolor)
    df["Iris-virginica"] = df["Class"].apply(encodervirginica)

    y = df[["Iris-setosa", "Iris-versicolor", "Iris-virginica"]].values
    X = df.drop(["Iris-setosa", "Iris-versicolor", "Iris-virginica", "Class"], axis=1).values

    nn = NerualNetwork(input_layer_size=4, hidden_layer_size=6, output_layer_size=3, learning_rate=0.1)
    nn.train(X, y, 1000)

    print("Accuracy on trainset is : " + str(nn.predict_and_calculate_accuracy(X, y)))

    predict_data=[4.8, 3.4, 1.6, 0.2]
    print("If an iris flower have X = {} , It is an {}".format(str(predict_data),nn.predict_one(predict_data)))
