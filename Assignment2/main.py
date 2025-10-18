import numpy as np
import pickle as pkl
import pandas as pd


def load_data(train_file, test_file):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    with open(train_file, "rb") as fp:
        train = pkl.load(fp)
    with open(test_file, "rb") as fp:
        test = pkl.load(fp)
    for image, label in train:
        train_data.append(image.flatten())
        train_labels.append(label)
    for image, label in test:
        test_data.append(image.flatten())
        test_labels.append(label)
    return train_data, train_labels, test_data, test_labels


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred))


def forward_propagation(X, W, b):
    z = np.dot(X, W) + b
    y_pred = softmax(z)
    y_pred_class = np.argmax(y_pred, axis=1)
    return z, y_pred, y_pred_class


def training(x_train, y_train, learning_rate=0.05, epochs=200):
    losses = []
    m, n_features = x_train.shape
    n_classes = 10
    W = np.random.randn(n_features, n_classes) * np.sqrt(2.0 / (n_features + n_classes))
    b = np.zeros(n_classes)
    y_train_encoded = np.eye(n_classes)[y_train]
    for epoch in range(epochs):
        z, y_pred, y_pred_class = forward_propagation(x_train, W, b)
        loss = cross_entropy_loss(y_pred, y_train_encoded)
        losses.append(loss)
        gradient = y_train_encoded - y_pred
        dW = np.dot(x_train.T, gradient)
        W += learning_rate * dW
        db = np.sum(gradient, axis=0)
        b += learning_rate * db
        if (epoch + 1) % 10 == 0:
            accuracy = np.mean(y_pred_class == y_train)
            print(f"epoch:{epoch + 1}, loss:{loss}, accuracy:{accuracy * 100}")
    return W, b, losses


def predict(X, W, b):
    _, _, y_pred_class = forward_propagation(X, W, b)
    return y_pred_class


def evaluate(x_test, y_test, W, b):
    _, y_pred_probs, y_pred_class = forward_propagation(x_test, W, b)
    y_test_encoded = np.eye(10)[y_test]
    accuracy = np.mean(y_pred_class == y_test)
    loss = cross_entropy_loss(y_pred_probs, y_test_encoded)
    return accuracy, loss


if __name__ == "__main__":
    train_file = "extended_mnist_train.pkl"
    test_file = "extended_mnist_test.pkl"
    train_data, train_labels, test_data, test_labels = load_data(train_file, test_file)
    x_train = np.array(train_data)
    y_train = np.array(train_labels)
    x_test = np.array(test_data)
    y_test = np.array(test_labels)
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-10
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    W, b, losses = training(x_train, y_train)
    test_accuracy, test_loss = evaluate(x_test, y_test, W, b)
    print(f"test_accuracy:{test_accuracy * 100}")
    print(f"test_loss:{test_loss}")
    predictions = predict(x_test, W, b)
    predictions_csv = {
        "ID": list(range(len(predictions))),
        "target": predictions.tolist(),
    }
    df = pd.DataFrame(predictions_csv)
    df.to_csv("submission.csv", index=False)
