import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


class PulseSequenceDataset(Dataset):

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class PulseLSTM(nn.Module):

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.30,
        output_dim: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden).squeeze(-1)


class LSTMLayer:

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.30,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        patience: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.model_: Optional[PulseLSTM] = None
        self.train_losses_: list = []
        self.val_losses_: list = []

    def _build_sequences(
        self, Phi_history: np.ndarray, seq_len: int = 6
    ) -> np.ndarray:
        n, d = Phi_history.shape
        if n < seq_len:
            pad = np.zeros((seq_len - n, d))
            Phi_history = np.vstack([pad, Phi_history])
        sequences = []
        for i in range(len(Phi_history) - seq_len + 1):
            sequences.append(Phi_history[i: i + seq_len])
        return np.array(sequences)

    def fit(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        X_val_seq: np.ndarray,
        y_val: np.ndarray,
    ) -> "LSTMLayer":
        torch.manual_seed(self.random_state)
        self.model_ = PulseLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        train_ds = PulseSequenceDataset(X_seq, y)
        val_ds = PulseSequenceDataset(X_val_seq, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_val_loss = np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = self.model_(X_batch)
                    val_loss += criterion(pred, y_batch).item()

            avg_train = epoch_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            self.train_losses_.append(avg_train)
            self.val_losses_.append(avg_val)
            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit first.")
        self.model_.eval()
        tensor = torch.FloatTensor(X_seq).to(self.device)
        with torch.no_grad():
            proba = self.model_(tensor).cpu().numpy()
        return proba

    def evaluate(self, X_seq: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        proba = self.predict_proba(X_seq)
        return {"auc": float(roc_auc_score(y, proba))}

    def save(self, path: str) -> None:
        torch.save({"model_state": self.model_.state_dict(), "config": self.__dict__}, path)