from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score

from ..data_preprocessing.eeg_data_preprocessing import preprocess_eeg_data
from ..models.eeg_model import SimpleEEGModel


def train_eeg_model(num_epochs=20, batch_size=64, learning_rate=0.001):
    """Train the app-compatible EEG classifier on the bundled EEG CSV dataset."""
    print("Training EEG model...")

    X_train, X_test, y_train, y_test = preprocess_eeg_data()

    train_dataset = TensorDataset(
        torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(np.asarray(X_test, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_test, dtype=np.int64)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(np.unique(y_train))
    model = SimpleEEGModel(num_classes=num_classes, input_length=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = len(y_train) / (num_classes * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for eeg_data, labels in train_loader:
            eeg_data, labels = eeg_data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {running_loss / total:.4f}, "
            f"Train Accuracy: {correct / total:.4f}"
        )

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for eeg_data, labels in test_loader:
            outputs = model(eeg_data.to(device))
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

    print("\nTest classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")

    output_path = Path("src/prediction/eeg_model_weights.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_path)
    print(f"EEG model weights saved to {output_path}")


if __name__ == "__main__":
    train_eeg_model()
