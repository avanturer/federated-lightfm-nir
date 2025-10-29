from typing import Dict, Tuple, List
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from .ncf_model import NeuralCF
from .data import load_movielens_implicit, build_id_maps, make_dataloader, split_clients
from .dp import clip_by_global_l2, add_gaussian_noise


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: optim.Optimizer) -> float:
    model.train()
    criterion = nn.BCELoss()
    total_loss = 0.0
    for users, items, labels in loader:
        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(users, items)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * users.shape[0]
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    with torch.no_grad():
        for users, items, labels in loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)
            preds = model(users, items)
            loss = criterion(preds, labels)
            total_loss += float(loss.item()) * users.shape[0]
    return total_loss / len(loader.dataset)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def set_parameters(model: nn.Module, params: List[np.ndarray]) -> None:
    sd = model.state_dict()
    new_sd = {}
    for (k, v), np_val in zip(sd.items(), params):
        new_sd[k] = torch.tensor(np_val)
    model.load_state_dict(new_sd, strict=False)


class RecsysClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, df_client, user_map, item_map, device: torch.device) -> None:
        self.client_id = client_id
        self.user_map = user_map
        self.item_map = item_map
        self.device = device
        # Stable split: 90% train, 10% val
        df_shuffled = df_client.sample(frac=1.0, random_state=client_id).reset_index(drop=True)
        split_idx = int(0.9 * len(df_shuffled))
        df_train = df_shuffled.iloc[:split_idx]
        df_val = df_shuffled.iloc[split_idx:]
        self.train_loader = make_dataloader(df_train, user_map, item_map, batch_size=1024)
        self.val_loader = make_dataloader(df_val, user_map, item_map, batch_size=2048)
        self.model = NeuralCF(num_users=len(user_map), num_items=len(item_map)).to(device)

    def get_parameters(self, config: Dict[str, str]):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Load global parameters (before)
        set_parameters(self.model, parameters)
        epochs = int(config.get("epochs", 1))
        dp_clip_norm = float(config.get("dp_clip_norm", 0.0))
        dp_sigma = float(config.get("dp_sigma", 0.0))
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Snapshot BEFORE params as tensors (same order as state_dict.values())
        before_tensors = [p.detach().clone() for p in self.model.state_dict().values()]

        for _ in range(epochs):
            train_one_epoch(self.model, self.train_loader, self.device, optimizer)

        # AFTER params
        after_tensors = [p.detach().clone() for p in self.model.state_dict().values()]
        # Compute delta = after - before
        deltas = [a - b for a, b in zip(after_tensors, before_tensors)]
        # Apply DP on delta
        deltas = clip_by_global_l2(deltas, dp_clip_norm) if dp_clip_norm and dp_clip_norm > 0 else deltas
        deltas = add_gaussian_noise(deltas, dp_sigma) if dp_sigma and dp_sigma > 0 else deltas
        # Build final params = before + noisy_delta
        final_params = [b + d for b, d in zip(before_tensors, deltas)]
        np_params = [p.cpu().numpy() for p in final_params]
        return np_params, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = evaluate(self.model, self.val_loader, self.device)
        return float(loss), len(self.val_loader.dataset), {"loss": float(loss)}


def start_client(client_id: int, server_address: str = "0.0.0.0:8080") -> None:
    df_all = load_movielens_implicit()
    clients = split_clients(df_all, n_clients=3)
    user_map, item_map = build_id_maps(df_all)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = RecsysClient(client_id=client_id, df_client=clients[client_id % len(clients)], user_map=user_map, item_map=item_map, device=device)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    cid = int(os.environ.get("CLIENT_ID", "0"))
    start_client(cid)


