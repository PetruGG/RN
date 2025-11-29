import numpy as np
import pickle as pkl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = get_device()
print(f"Using device: {device}")


class MNISTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


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
        train_data.append(image)
        train_labels.append(label)
    for image, label in test:
        test_data.append(image)
        test_labels.append(label)
    return train_data, train_labels, test_data, test_labels


class MNISTNet(nn.Module):
    def __init__(self, input_size=784, hidden_neurons=512, num_classes=10, dropout_rate=0.2):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons // 2)
        self.fc3 = nn.Linear(hidden_neurons // 2, hidden_neurons // 4)
        self.fc4 = nn.Linear(hidden_neurons // 4, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


@torch.inference_mode()
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100.0 * correct / total
    return val_loss, val_accuracy


def training(x_train, y_train, x_val, y_val, learning_rate=0.001, epochs=250, patience=50,
             delta_improvement=0.00005, dropout_rate=0.2, hidden_neurons=512, batch_size=64,
             weight_decay=1e-5, optimizer_type='adam'):
    train_dataset = MNISTDataset(x_train, y_train)
    val_dataset = MNISTDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MNISTNet(
        input_size=x_train.shape[1],
        hidden_neurons=hidden_neurons,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    with tqdm(range(epochs), desc=f"Training with {optimizer_type}") as pbar:
        for epoch in pbar:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            losses.append(train_loss)

            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            scheduler.step(val_loss)

            if val_loss < best_val_loss - delta_improvement:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Accuracy': f'{val_accuracy:.2f}%',
                'LR': f'{current_lr:.6f}'
            })

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, losses, val_losses, val_accuracies


@torch.inference_mode()
def predict(model, test_loader, device):
    model.eval()
    all_predictions = []

    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        _, predictions = torch.max(outputs, 1)
        all_predictions.extend(predictions.cpu().numpy())

    return all_predictions


if __name__ == '__main__':
    train_file = "extended_mnist_train.pkl"
    test_file = "extended_mnist_test.pkl"
    train_data, train_labels, test_data, test_labels = load_data(train_file, test_file)
    x_train = np.array(train_data)
    y_train = np.array(train_labels)
    x_test = np.array(test_data)
    y_test = np.array(test_labels)

    if np.any(y_test == -1) or len(y_test) == 0:
        y_test_dummy = np.zeros(len(x_test), dtype=np.int64)
    else:
        y_test_dummy = y_test

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-10
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Training set size: {x_train_split.shape[0]}")
    print(f"Validation set size: {x_val.shape[0]}")
    print(f"Test set size: {x_test.shape[0]}")

    test_dataset = MNISTDataset(x_test, y_test_dummy)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    optimizer_type = 'rmsprop'

    model, losses, val_losses, val_accuracies = training(
        x_train_split, y_train_split, x_val, y_val,
        learning_rate=0.001,
        epochs=120,
        patience=30,
        dropout_rate=0.2,
        hidden_neurons=512,
        batch_size=128,
        weight_decay=1e-4,
        optimizer_type=optimizer_type
    )
    predictions = predict(model, test_loader, device)
    predictions_csv = {
        "ID": list(range(len(predictions))),
        "target": predictions
    }
    df = pd.DataFrame(predictions_csv)
    df.to_csv("submission.csv", index=False)
