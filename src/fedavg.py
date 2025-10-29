from .lightfm_model import build_interaction_matrix_with_dataset
import numpy as np
from .bandit import EpsilonGreedyBandit
from lightfm.evaluation import precision_at_k
import collections
from lightfm import LightFM

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

def _clip_array_by_l2_norm(arr, max_norm):
    if max_norm is None or max_norm <= 0:
        return arr
    norm = np.linalg.norm(arr.ravel(), ord=2)
    if norm == 0 or norm <= max_norm:
        return arr
    scale = max_norm / norm
    return arr * scale

def average_weights(weights_list, clip_norm: float | None = None, noise_sigma: float | None = None, rng: np.random.Generator | None = None, client_weights: np.ndarray | None = None):
    n_types = len(weights_list[0])
    avg = []
    for i in range(n_types):
        # Находим максимальную форму среди всех клиентов
        shapes = [w[i].shape for w in weights_list]
        max_shape = tuple(np.max(shapes, axis=0))
        padded = [pad_to_shape(w[i], max_shape) for w in weights_list]
        # Optional per-client clipping before averaging
        if clip_norm is not None and clip_norm > 0:
            padded = [_clip_array_by_l2_norm(w, clip_norm) for w in padded]
        stacked = np.stack(padded, axis=0)
        if client_weights is not None:
            w = client_weights.astype(float)
            w = w / (np.sum(w) + 1e-12)
            mean_w = np.tensordot(w, stacked, axes=(0, 0))
        else:
            mean_w = np.mean(stacked, axis=0)
        if noise_sigma is not None and noise_sigma > 0:
            if rng is None:
                rng = np.random.default_rng()
            mean_w = mean_w + rng.normal(loc=0.0, scale=noise_sigma, size=mean_w.shape)
        avg.append(mean_w)
    return avg

def federated_training(
    clients_data,
    dataset,
    rounds=3,
    epoch_arms=[3, 5, 10],
    epsilon=0.2,
    dp_clip_norm: float | None = None,
    dp_noise_sigma: float | None = None,
    random_seed: int | None = 42,
    test_interactions=None,
    # aggregation
    use_weighted_agg: bool = True,
    # model params
    no_components: int = 256,
    loss: str = 'warp',
    learning_schedule: str = 'adagrad',
    learning_rate: float = 0.08,
    item_alpha: float = 5e-7,
    user_alpha: float = 5e-7,
    num_threads_alloc: int = 8,
    num_threads_fit: int = 8,
):
    n_clients = len(clients_data)
    rng = np.random.default_rng(random_seed)
    bandits = [EpsilonGreedyBandit(arms=epoch_arms, epsilon=epsilon, rng=np.random.default_rng((random_seed or 0) + 1000 + i)) for i in range(n_clients)]
    # Для сбора статистики по arms
    arm_stats = [collections.defaultdict(list) for _ in range(n_clients)]
    # История метрик по раундам
    history = {
        "round_global_precision": [],
        "round_client_precisions": [],  # список списков по клиентам
        "round_client_epochs": []       # список списков выбранных эпох по клиентам
    }
    global_weights = None
    # Параметры epsilon decay: линейно от epsilon до max(0.05, epsilon/2)
    eps_start = float(epsilon)
    eps_end = max(0.05, eps_start / 2.0)
    for rnd in range(rounds):
        print(f"\n=== FedAvg: Раунд {rnd+1}/{rounds} ===")
        # Обновляем epsilon для всех bandit (decay по раундам)
        frac = rnd / max(1, rounds - 1)
        eps_now = eps_start + (eps_end - eps_start) * frac
        for b in bandits:
            b.set_epsilon(eps_now)
        models = []
        client_precisions = []
        client_epochs_chosen = []
        client_train_sizes = []
        for idx, client in enumerate(clients_data):
            bandit = bandits[idx]
            arm = bandit.select_arm()
            epochs = epoch_arms[arm]
            print(f"  [CLIENT {idx+1}] Выбранное число эпох: {epochs}")
            train_inter = build_interaction_matrix_with_dataset(client['train'], dataset)
            # Создаём локальную модель, инициализируем глобальными весами ДО обучения
            model = LightFM(no_components=no_components, loss=loss, learning_schedule=learning_schedule, item_alpha=item_alpha, user_alpha=user_alpha, learning_rate=learning_rate, random_state=42)
            # Аллоцируем параметры по форме матрицы взаимодействий
            model.fit(train_inter, epochs=0, num_threads=num_threads_alloc)
            if global_weights is not None:
                set_model_weights(model, global_weights)
            # Теперь обучаем заданное число эпох продолжая с глобальных весов
            model.fit(train_inter, epochs=epochs, num_threads=num_threads_fit)
            val_inter = build_interaction_matrix_with_dataset(client['val'], dataset)
            prec = precision_at_k(model, val_inter, k=5).mean()
            print(f"    Precision@5 на валидации: {prec:.4f}")
            bandit.update(arm, prec)
            arm_stats[idx][epochs].append(prec)
            client_precisions.append(float(prec))
            client_epochs_chosen.append(int(epochs))
            client_train_sizes.append(int(len(client['train'])))
            models.append(model)
        weights_list = [get_model_weights(m) for m in models]
        if use_weighted_agg:
            client_weights = np.array(client_train_sizes, dtype=float)
            norm_w = client_weights / (np.sum(client_weights) + 1e-12)
            print("  [AGG] Размер обучающей выборки по клиентам:", client_train_sizes)
            print("  [AGG] Нормированные веса агрегации:", [f"{w:.3f}" for w in norm_w])
            global_weights = average_weights(weights_list, clip_norm=dp_clip_norm, noise_sigma=dp_noise_sigma, rng=rng, client_weights=client_weights)
        else:
            print("  [AGG] Равноправное усреднение без весов клиентов")
            global_weights = average_weights(weights_list, clip_norm=dp_clip_norm, noise_sigma=dp_noise_sigma, rng=rng, client_weights=None)
        print("=== FedAvg завершён. Модели синхронизированы. ===")
        # Оценка глобальной модели на тесте по окончании раунда (если предоставлено)
        if test_interactions is not None and global_weights is not None:
            # Чистая оценка: новый экземпляр с той же конфигурацией, аллокация и установка весов
            tmp_global = LightFM(no_components=no_components, loss=loss, learning_schedule=learning_schedule, item_alpha=item_alpha, user_alpha=user_alpha, learning_rate=learning_rate, random_state=42)
            tmp_global.fit(test_interactions, epochs=0, num_threads=num_threads_alloc)
            set_model_weights(tmp_global, global_weights)
            round_prec = precision_at_k(tmp_global, test_interactions, k=5).mean()
            history["round_global_precision"].append(float(round_prec))
        else:
            history["round_global_precision"].append(None)
        history["round_client_precisions"].append(client_precisions)
        history["round_client_epochs"].append(client_epochs_chosen)
    # Возвращаем финальную глобальную модель
    final_model = models[0]
    set_model_weights(final_model, global_weights)
    print("\n=== Статистика bandit по клиентам ===")
    for idx, stats in enumerate(arm_stats):
        print(f"[CLIENT {idx+1}] Arms usage:")
        for arm in epoch_arms:
            rewards = stats[arm]
            count = len(rewards)
            if rewards:
                avg_reward_str = f"{np.mean(rewards):.4f}"
            else:
                avg_reward_str = "N/A"
            print(f"  Эпох: {arm} | Выбран: {count} раз | Средний Precision@5: {avg_reward_str}")
    return final_model, history