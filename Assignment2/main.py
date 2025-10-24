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


def forward_propagation(X, W, b, dropout_mask=None):
    if dropout_mask is not None:
        X = X * dropout_mask
    z = np.dot(X, W) + b
    y_pred = softmax(z)
    y_pred_class = np.argmax(y_pred, axis=1)
    return z, y_pred, y_pred_class


def training(x_train, y_train, x_val, y_val, learning_rate=0.075, epochs=250, patience=25, delta_improvement=0.0002,
             dropout_rate=0.3, momentum=0.9):
    losses = []
    val_losses = []
    m, n_features = x_train.shape
    n_classes = 10

    W = np.random.randn(n_features, n_classes) * np.sqrt(2.0 / (n_features + n_classes))
    b = np.zeros(n_classes)

    v_dW = np.zeros_like(W)
    v_db = np.zeros_like(b)

    y_train_encoded = np.eye(n_classes)[y_train]
    y_val_encoded = np.eye(n_classes)[y_val]
    best_val_loss = float('inf')
    best_W = W.copy()
    best_b = b.copy()
    patience_counter = 0

    for epoch in range(epochs):
        dropout_mask = (np.random.rand(*x_train.shape) > dropout_rate) / (1 - dropout_rate)

        z, y_pred, y_pred_class = forward_propagation(x_train, W, b, dropout_mask)
        loss = cross_entropy_loss(y_pred, y_train_encoded)
        losses.append(loss)

        _, y_val_pred, _ = forward_propagation(x_val, W, b)
        val_loss = cross_entropy_loss(y_val_pred, y_val_encoded)
        val_losses.append(val_loss)

        gradient = y_train_encoded - y_pred
        dW = np.dot(x_train.T, gradient)
        db = np.sum(gradient, axis=0)

        v_dW = momentum * v_dW + learning_rate * dW
        v_db = momentum * v_db + learning_rate * db

        W += v_dW
        b += v_db

        if val_loss < best_val_loss - delta_improvement:
            best_val_loss = val_loss
            best_W = W.copy()
            best_b = b.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            accuracy = np.mean(y_pred_class == y_train)
            val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == y_val)
            print(f"epoch:{epoch + 1}, loss:{loss:.4f}, accuracy:{accuracy * 100:.2f}%, "
                  f"val_loss:{val_loss:.4f}, val_accuracy:{val_accuracy * 100:.2f}%")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
            break

    return best_W, best_b, losses, val_losses


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

    split_idx = int(0.8 * len(x_train))
    x_train_split = x_train[:split_idx]
    y_train_split = y_train[:split_idx]
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]

    W, b, losses, val_losses = training(x_train_split, y_train_split, x_val, y_val)
    test_accuracy, test_loss = evaluate(x_test, y_test, W, b)
    print(f"test_accuracy:{test_accuracy * 100:.2f}%")
    print(f"test_loss:{test_loss:.4f}")

    predictions = predict(x_test, W, b)
    predictions_csv = {
        "ID": list(range(len(predictions))),
        "target": predictions.tolist(),
    }
    df = pd.DataFrame(predictions_csv)
    df.to_csv("submission.csv", index=False)
