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


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]


def forward_step(X, W1, b1, W2, b2, dropout_mask=None):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    if dropout_mask is not None:
        a1 = a1 * dropout_mask

    z2 = np.dot(a1, W2) + b2

    y_pred = softmax(z2)
    y_pred_class = np.argmax(y_pred, axis=1)

    return z1, a1, z2, y_pred, y_pred_class


def backward_step(X_batch, y_batch, z1, a1, z2, W2, dropout_mask=None):
    batch_size_current = X_batch.shape[0]
    n_classes = W2.shape[1]

    softmax_probs = softmax(z2)

    y_one_hot = np.eye(n_classes)[y_batch]

    gradient_z2 = (softmax_probs - y_one_hot) / batch_size_current

    relu_grad = relu_derivative(z1)
    if dropout_mask is not None:
        relu_grad = relu_grad * dropout_mask

    gradient_z1 = np.dot(gradient_z2, W2.T) * relu_grad

    dW2 = np.dot(a1.T, gradient_z2)
    db2 = np.sum(gradient_z2, axis=0)
    dW1 = np.dot(X_batch.T, gradient_z1)
    db1 = np.sum(gradient_z1, axis=0)

    return dW1, db1, dW2, db2


def training(x_train, y_train, x_val, y_val, learning_rate=0.015, epochs=250, patience=50, delta_improvement=0.00005,
             dropout_rate=0.2, momentum=0.9, hidden_neurons=100, batch_size=64, weight_decay=1e-5):
    losses = []
    val_losses = []
    m, n_features = x_train.shape
    n_classes = 10

    W1 = np.random.randn(n_features, hidden_neurons) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(hidden_neurons)
    W2 = np.random.randn(hidden_neurons, n_classes) * np.sqrt(2.0 / (hidden_neurons + n_classes))
    b2 = np.zeros(n_classes)

    v_dW1 = np.zeros_like(W1)
    v_db1 = np.zeros_like(b1)
    v_dW2 = np.zeros_like(W2)
    v_db2 = np.zeros_like(b2)

    y_val_encoded = np.eye(n_classes)[y_val]

    best_val_loss = float('inf')
    best_W1 = W1.copy()
    best_b1 = b1.copy()
    best_W2 = W2.copy()
    best_b2 = b2.copy()
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        n_samples = x_train.shape[0]
        indices = np.random.permutation(n_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        y_train_encoded = np.eye(n_classes)[y_train_shuffled]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            y_batch_encoded = y_train_encoded[start:end]

            dropout_mask = (np.random.rand(X_batch.shape[0], hidden_neurons) > dropout_rate) / (1 - dropout_rate)

            z1, a1, z2, y_pred, y_pred_class = forward_step(X_batch, W1, b1, W2, b2, dropout_mask)

            batch_loss = cross_entropy_loss(y_pred, y_batch_encoded)
            epoch_loss += batch_loss
            n_batches += 1

            dW1, db1, dW2, db2 = backward_step(X_batch, y_batch, z1, a1, z2, W2, dropout_mask)

            dW1 += weight_decay * W1
            dW2 += weight_decay * W2

            v_dW2 = momentum * v_dW2 + learning_rate * dW2
            v_db2 = momentum * v_db2 + learning_rate * db2
            v_dW1 = momentum * v_dW1 + learning_rate * dW1
            v_db1 = momentum * v_db1 + learning_rate * db1

            W2 -= v_dW2
            b2 -= v_db2
            W1 -= v_dW1
            b1 -= v_db1

        avg_epoch_loss = epoch_loss / n_batches
        losses.append(avg_epoch_loss)

        _, _, z2_val, y_val_pred, _ = forward_step(x_val, W1, b1, W2, b2)
        val_loss = cross_entropy_loss(y_val_pred, y_val_encoded)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - delta_improvement:
            best_val_loss = val_loss
            best_W1 = W1.copy()
            best_b1 = b1.copy()
            best_W2 = W2.copy()
            best_b2 = b2.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            _, _, _, _, y_train_pred_class = forward_step(x_train, W1, b1, W2, b2)
            accuracy = np.mean(y_train_pred_class == y_train)
            val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == y_val)
            print(f"epoch:{epoch + 1}, loss:{avg_epoch_loss:.4f}, accuracy:{accuracy * 100:.2f}%, "
                  f"val_loss:{val_loss:.4f}, val_accuracy:{val_accuracy * 100:.2f}%")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
            break

    return best_W1, best_b1, best_W2, best_b2, losses, val_losses


def predict(X, W1, b1, W2, b2):
    _, _, _, _, y_pred_class = forward_step(X, W1, b1, W2, b2)
    return y_pred_class


def evaluate(x_test, y_test, W1, b1, W2, b2):
    _, _, z2_test, y_pred_probs, y_pred_class = forward_step(x_test, W1, b1, W2, b2)
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

    W1, b1, W2, b2, losses, val_losses = training(x_train_split, y_train_split, x_val, y_val)
    test_accuracy, test_loss = evaluate(x_test, y_test, W1, b1, W2, b2)
    print(f"test_accuracy:{test_accuracy * 100:.2f}%")
    print(f"test_loss:{test_loss:.4f}")

    predictions = predict(x_test, W1, b1, W2, b2)
    predictions_csv = {
        "ID": list(range(len(predictions))),
        "target": predictions.tolist(),
    }
    df = pd.DataFrame(predictions_csv)
    df.to_csv("submission.csv", index=False)
