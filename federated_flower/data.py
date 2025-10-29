from typing import Tuple, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_movielens_implicit() -> pd.DataFrame:
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    df = pd.read_csv(url, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])  # implicit
    # Convert to implicit feedback (1 for any interaction)
    df = df[["user_id", "item_id"]].drop_duplicates()
    return df


def split_clients(df: pd.DataFrame, n_clients: int = 3, seed: int = 42) -> List[pd.DataFrame]:
    return list(np.array_split(df.sample(frac=1.0, random_state=seed), n_clients))


class InteractionsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user_id_map: dict[int, int], item_id_map: dict[int, int]) -> None:
        self.users = torch.tensor([user_id_map[u] for u in df["user_id"].astype(int).tolist()], dtype=torch.long)
        self.items = torch.tensor([item_id_map[i] for i in df["item_id"].astype(int).tolist()], dtype=torch.long)
        self.labels = torch.ones(len(self.users), dtype=torch.float32)

    def __len__(self) -> int:
        return self.users.shape[0]

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.labels[idx]


def build_id_maps(df_all: pd.DataFrame) -> Tuple[dict[int, int], dict[int, int]]:
    users = sorted(df_all["user_id"].astype(int).unique())
    items = sorted(df_all["item_id"].astype(int).unique())
    user_id_map = {uid: i for i, uid in enumerate(users)}
    item_id_map = {iid: i for i, iid in enumerate(items)}
    return user_id_map, item_id_map


def make_dataloader(df: pd.DataFrame, user_map: dict[int, int], item_map: dict[int, int], batch_size: int = 512) -> DataLoader:
    dataset = InteractionsDataset(df, user_map, item_map)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


