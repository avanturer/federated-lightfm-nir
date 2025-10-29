import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_movielens():
    # Для быстрого старта используем небольшой поднабор MovieLens 100k
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    df = pd.read_csv(url, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    return df

def split_train_test(df, test_size=0.2, random_state=42):
    # Хронологический сплит без «подглядывания»: ранние записи в train, поздние в test
    df_sorted = df.sort_values('timestamp')
    n_test = int(len(df_sorted) * test_size)
    test = df_sorted.iloc[-n_test:]
    train = df_sorted.iloc[:-n_test]
    return train, test

def get_full_user_item_lists(df):
    users = sorted(df['user_id'].unique())
    items = sorted(df['item_id'].unique())
    return users, items

def split_data_among_clients(df, n_clients=3, all_data=None):
    # Перемешиваем только train (это допустимо), но без смешивания с более поздним тестом
    splits = np.array_split(df.sample(frac=1, random_state=42), n_clients)
    # users/items теперь собираем из all_data (train+test), если передано
    if all_data is not None:
        users, items = get_full_user_item_lists(all_data)
    else:
        users, items = get_full_user_item_lists(df)
    return splits, users, items

def split_client_train_val(df, val_size=0.1, random_state=42):
    # Хронологический сплит валидации внутри клиентской выборки
    df_sorted = df.sort_values('timestamp')
    n_val = int(len(df_sorted) * val_size)
    val = df_sorted.iloc[-n_val:]
    train = df_sorted.iloc[:-n_val]
    return train, val