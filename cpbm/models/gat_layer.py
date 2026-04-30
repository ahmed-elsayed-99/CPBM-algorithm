import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import roc_auc_score
from typing import Optional, Dict, List, Tuple
import networkx as nx
import warnings
warnings.filterwarnings("ignore")


class PulseGAT(nn.Module):

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 32,
        num_heads: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_channels,
            heads=num_heads, dropout=dropout, concat=True
        )
        self.conv2 = GATConv(
            hidden_channels * num_heads, hidden_channels,
            heads=1, dropout=dropout, concat=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


class GATLayer:

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 32,
        num_heads: int = 2,
        dropout: float = 0.30,
        epochs: int = 60,
        learning_rate: float = 5e-3,
        patience: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.model_: Optional[PulseGAT] = None
        self.train_losses_: list = []

    @staticmethod
    def graph_to_edge_index(G: nx.Graph) -> torch.Tensor:
        edges = list(G.edges())
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        src_all = src + dst
        dst_all = dst + src
        return torch.tensor([src_all, dst_all], dtype=torch.long)

    def build_pyg_data(
        self,
        Phi: np.ndarray,
        G: nx.Graph,
        labels: np.ndarray,
    ) -> Data:
        x = torch.FloatTensor(Phi)
        edge_index = self.graph_to_edge_index(G)
        y = torch.FloatTensor(labels)
        return Data(x=x, edge_index=edge_index, y=y)

    def fit(
        self,
        data_train: Data,
        data_val: Data,
    ) -> "GATLayer":
        torch.manual_seed(self.random_state)
        self.model_ = PulseGAT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        best_val_loss = np.inf
        patience_counter = 0
        best_state = None

        data_train = data_train.to(self.device)
        data_val = data_val.to(self.device)

        for epoch in range(self.epochs):
            self.model_.train()
            optimizer.zero_grad()
            out = self.model_(data_train.x, data_train.edge_index)
            loss = criterion(out, data_train.y)
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

            self.model_.eval()
            with torch.no_grad():
                val_out = self.model_(data_val.x, data_val.edge_index)
                val_loss = criterion(val_out, data_val.y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, data: "Data") -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit first.")
        self.model_.eval()
        data = data.to(self.device)
        with torch.no_grad():
            proba = self.model_(data.x, data.edge_index).cpu().numpy()
        return proba

    def evaluate(self, data: "Data") -> Dict[str, float]:
        proba = self.predict_proba(data)
        y = data.y.cpu().numpy()
        return {"auc": float(roc_auc_score(y, proba))}