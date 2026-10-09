"""Learning tasks adapted from the two original coursework notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HousingTask:
    """Small tabular regression benchmark from the original search notebook.

    The original eight-row example is intentionally retained as a transparent
    smoke-test task. It is not a substitute for a real housing dataset and is
    reported as a pedagogical benchmark only.
    """

    def __post_init__(self) -> None:
        self.locations = np.array(
            ["Downtown", "Suburb", "Countryside", "Downtown", "Suburb", "Countryside", "Downtown", "Suburb"]
        )
        self.features = np.array(
            [
                [1500, 3, 2, 1, 5],
                [2000, 4, 3, 1, 10],
                [2500, 4, 2, 0, 15],
                [1800, 3, 2, 1, 3],
                [2200, 5, 3, 1, 8],
                [2700, 4, 2, 0, 20],
                [1700, 3, 2, 1, 4],
                [2100, 4, 3, 1, 7],
            ],
            dtype=float,
        )
        self.targets = np.array([400000, 350000, 280000, 450000, 320000, 260000, 420000, 340000], dtype=float)
        self.train_indices = np.array([0, 1, 2, 3, 4, 6])
        self.test_indices = np.array([5, 7])
        self.categories = ["Countryside", "Downtown", "Suburb"]

    def _matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        encoded = np.column_stack([(self.locations == category).astype(float) for category in self.categories])
        train_values = self.features[self.train_indices]
        mean = train_values.mean(axis=0)
        scale = train_values.std(axis=0)
        scale[scale == 0] = 1.0
        normalized = (self.features - mean) / scale
        x = np.column_stack([np.ones(len(self.features)), encoded, normalized])
        y = self.targets / 100000.0
        return x[self.train_indices], x[self.test_indices], y[self.train_indices], y[self.test_indices]

    def evaluate(self, config: dict[str, Any]) -> float:
        x_train, x_test, y_train, y_test = self._matrix()
        theta = np.zeros(x_train.shape[1])
        learning_rate = float(config["learning_rate"])
        l2 = float(config["l2"])
        for _ in range(int(config["epochs"])):
            residual = x_train @ theta - y_train
            gradient = (2.0 / len(y_train)) * (x_train.T @ residual)
            gradient[1:] += 2.0 * l2 * theta[1:]
            theta -= learning_rate * gradient
        prediction = x_test @ theta
        return float(np.mean((prediction - y_test) ** 2))

    def summary(self, config: dict[str, Any]) -> dict[str, Any]:
        x_train, x_test, y_train, y_test = self._matrix()
        theta = np.zeros(x_train.shape[1])
        learning_rate = float(config["learning_rate"])
        l2 = float(config["l2"])
        for _ in range(int(config["epochs"])):
            residual = x_train @ theta - y_train
            gradient = (2.0 / len(y_train)) * (x_train.T @ residual)
            gradient[1:] += 2.0 * l2 * theta[1:]
            theta -= learning_rate * gradient
        test_prediction = x_test @ theta
        return {
            "config": config,
            "test_mse_scaled": float(np.mean((test_prediction - y_test) ** 2)),
            "test_rmse_dollars": float(np.sqrt(np.mean((test_prediction - y_test) ** 2)) * 100000),
            "theta": theta.round(6).tolist(),
            "note": "Eight-row pedagogical benchmark; not a population-level housing evaluation.",
        }


class MNISTTask:
    """PyTorch task adapted from the original MNIST MLP notebook.

    Imports are delayed so housing search and documentation tooling can run
    without downloading MNIST or importing the optional PyTorch stack.
    """

    def evaluate(self, config: dict[str, Any]) -> float:
        _, accuracy = self.run(config, train_samples=int(config.get("train_samples", 10000)))
        return 1.0 - accuracy

    def run(self, config: dict[str, Any], train_samples: int = 10000) -> tuple[float, float]:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader, Subset

        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_data = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        train_count = min(train_samples, len(train_data) - 10000)
        train_set = Subset(train_data, list(range(train_count)))
        validation_set = Subset(train_data, list(range(train_count, train_count + 10000)))
        test_set = Subset(test_data, list(range(min(2000, len(test_data)))))
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28 * 28, int(config["hidden1"])),
                    nn.ReLU(),
                    nn.Dropout(float(config["dropout"])),
                    nn.Linear(int(config["hidden1"]), int(config["hidden2"])),
                    nn.ReLU(),
                    nn.Linear(int(config["hidden2"]), 10),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        model = MLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        criterion = nn.CrossEntropyLoss()

        def score(loader: DataLoader) -> tuple[float, float]:
            model.eval()
            total_loss = correct = total = 0.0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    total_loss += criterion(outputs, labels).item() * len(labels)
                    correct += float((outputs.argmax(1) == labels).sum())
                    total += len(labels)
            return total_loss / total, correct / total

        for _ in range(int(config["epochs"])):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
        validation_loss, _ = score(validation_loader)
        _, test_accuracy = score(test_loader)
        return validation_loss, test_accuracy
