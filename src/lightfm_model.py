from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np

def build_interaction_matrix(df, users, items):
    print(f"[DEBUG] build_interaction_matrix: users={len(users)}, items={len(items)}")
    dataset = Dataset()
    dataset.fit(users, items)
    interactions, _ = dataset.build_interactions([(row['user_id'], row['item_id']) for _, row in df.iterrows()])
    return interactions, dataset

def build_interaction_matrix_with_dataset(df, dataset):
    print(f"[DEBUG] build_interaction_matrix_with_dataset: df users={df['user_id'].nunique()}, items={df['item_id'].nunique()}")
    interactions, _ = dataset.build_interactions([(row['user_id'], row['item_id']) for _, row in df.iterrows()])
    return interactions

def train_lightfm(interactions, epochs=5, no_components=10, random_state=42):
    print(f"[DEBUG] train_lightfm: interactions shape = {interactions.shape}")
    if interactions.shape[0] == 0 or interactions.shape[1] == 0:
        print("[ERROR] Пустая матрица взаимодействий! Возвращаю None.")
        return None
    model = LightFM(no_components=no_components, random_state=random_state)
    model.fit(interactions, epochs=epochs, num_threads=1)
    return model 