from .lightfm_model import build_interaction_matrix_with_dataset, train_lightfm
import numpy as np
from .bandit import EpsilonGreedyBandit
from lightfm.evaluation import precision_at_k
import collections

def get_model_weights(model):
    return [
        model.user_embeddings.copy(),
        model.item_embeddings.copy(),
        model.user_biases.copy(),
        model.item_biases.copy()
    ]

def set_model_weights(model, weights):
    model.user_embeddings[:] = weights[0]
    model.item_embeddings[:] = weights[1]
    model.user_biases[:] = weights[2]
    model.item_biases[:] = weights[3]
    return model

def pad_to_shape(arr, shape):
    # Дополняет массив нулями до нужной формы
    pad_width = [(0, max(0, s - arr.shape[i])) for i, s in enumerate(shape)]
    return np.pad(arr, pad_width, mode='constant')

def average_weights(weights_list):
    n_types = len(weights_list[0])
    avg = []
    for i in range(n_types):
        # Находим максимальную форму среди всех клиентов
        shapes = [w[i].shape for w in weights_list]
        max_shape = tuple(np.max(shapes, axis=0))
        padded = [pad_to_shape(w[i], max_shape) for w in weights_list]
        stacked = np.stack(padded, axis=0)
        avg.append(np.mean(stacked, axis=0))
    return avg

def federated_training(clients_data, dataset, rounds=3, epoch_arms=[3, 5, 10], epsilon=0.2):
    n_clients = len(clients_data)
    bandits = [EpsilonGreedyBandit(arms=epoch_arms, epsilon=epsilon) for _ in range(n_clients)]
    # Для сбора статистики по arms
    arm_stats = [collections.defaultdict(list) for _ in range(n_clients)]
    global_weights = None
    for rnd in range(rounds):
        print(f"\n=== FedAvg: Раунд {rnd+1}/{rounds} ===")
        models = []
        for idx, client in enumerate(clients_data):
            bandit = bandits[idx]
            arm = bandit.select_arm()
            epochs = epoch_arms[arm]
            print(f"  [CLIENT {idx+1}] Выбранное число эпох: {epochs}")
            train_inter = build_interaction_matrix_with_dataset(client['train'], dataset)
            model = train_lightfm(train_inter, epochs=epochs)
            if global_weights is not None:
                set_model_weights(model, global_weights)
            val_inter = build_interaction_matrix_with_dataset(client['val'], dataset)
            prec = precision_at_k(model, val_inter, k=5).mean()
            print(f"    Precision@5 на валидации: {prec:.4f}")
            bandit.update(arm, prec)
            arm_stats[idx][epochs].append(prec)
            models.append(model)
        weights_list = [get_model_weights(m) for m in models]
        global_weights = average_weights(weights_list)
        print("=== FedAvg завершён. Модели синхронизированы. ===")
    # Возвращаем финальную глобальную модель
    final_model = models[0]
    set_model_weights(final_model, global_weights)
    print("\n=== Статистика bandit по клиентам ===")
    for idx, stats in enumerate(arm_stats):
        print(f"[CLIENT {idx+1}] Arms usage:")
        for arm in epoch_arms:
            rewards = stats[arm]
            count = len(rewards)
            avg_reward = np.mean(rewards) if rewards else 0.0
            print(f"  Эпох: {arm} | Выбран: {count} раз | Средний Precision@5: {avg_reward:.4f}")
    return final_model 