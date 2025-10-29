import torch
import torch.nn as nn


class NeuralCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        mlp_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_indices)
        v = self.item_embedding(item_indices)
        x = torch.cat([u, v], dim=-1)
        x = self.mlp(x)
        return self.sigmoid(x).squeeze(-1)


