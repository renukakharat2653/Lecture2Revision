
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pth")

torch.manual_seed(42)
np.random.seed(42)


class ScoreLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


def create_demo_data(count=1200):
    X, y = [], []

    for _ in range(count):
        start = np.random.uniform(20, 85)
        trend = np.random.uniform(-8, 8)
        scores = [
            np.clip(start + trend * i + np.random.normal(0, 5), 0, 100)
            for i in range(5)
        ]

        # Label 1 = needs more revision
        needs_revision = int(
            scores[-1] < 60 or
            np.mean(scores[-3:]) < 55
        )

        X.append(np.array(scores) / 100.0)
        y.append(needs_revision)

    return (
        torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y, dtype=torch.long)
    )


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = create_demo_data()
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=32,
        shuffle=True
    )

    model = ScoreLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(12):
        total_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/12 "
            f"- Loss: {total_loss / len(loader):.4f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nLSTM model saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
