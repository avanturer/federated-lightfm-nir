from .data_utils import load_movielens, split_data_among_clients, split_train_test, get_full_user_item_lists, split_client_train_val
from .fedavg import federated_training
from .lightfm_model import build_interaction_matrix_with_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from lightfm.evaluation import precision_at_k

if __name__ == "__main__":
    print("========== [FEDERATED LIGHTFM DEMO] ==========")
    print("[INFO] Загрузка и подготовка данных MovieLens...")
    data = load_movielens()
    print(f"[INFO] Всего записей: {len(data)} | Уникальных пользователей: {data['user_id'].nunique()} | Уникальных фильмов: {data['item_id'].nunique()}")

    print("[INFO] Делим данные на train/test...")
    train_df, test_df = split_train_test(data, test_size=0.2, random_state=42)
    print(f"[INFO] Train: {len(train_df)} записей | Test: {len(test_df)} записей")

    n_clients = 3
    print(f"[INFO] Разделение train-данных между {n_clients} клиентами...")
    raw_clients_data, users, items = split_data_among_clients(train_df, n_clients=n_clients, all_data=data)
    clients_data = []
    for i, cdata in enumerate(raw_clients_data):
        train_c, val_c = split_client_train_val(cdata, val_size=0.1, random_state=42)
        clients_data.append({'train': train_c, 'val': val_c})
        print(f"  [CLIENT {i+1}] Train: {len(train_c)} | Val: {len(val_c)} | Уникальных пользователей: {cdata['user_id'].nunique()} | Уникальных фильмов: {cdata['item_id'].nunique()}")

    # Собираем users/items из train+test для fit
    all_users, all_items = get_full_user_item_lists(data)

    print("[INFO] Создание общего Dataset для всех клиентов и теста...")
    from lightfm.data import Dataset
    dataset = Dataset()
    dataset.fit(all_users, all_items)
    train_interactions = build_interaction_matrix_with_dataset(train_df, dataset)
    test_interactions = build_interaction_matrix_with_dataset(test_df, dataset)

    print("[INFO] Запуск федеративного обучения (FedAvg)...")
    global_model, history = federated_training(
        clients_data,
        dataset,
        rounds=12,
        epoch_arms=[10, 20, 30, 40],
        epsilon=0.2,
        test_interactions=test_interactions
    )
    if global_model is None:
        print("[FATAL] Глобальная модель не создана. Проверьте корректность FedAvg и входных данных!")
        import sys; sys.exit(1)
    print("[INFO] FedAvg завершён. Модели синхронизированы.")
    print("[DEBUG] Тип global_model:", type(global_model))

    print("\n=== Оценка итоговой глобальной модели на тестовой выборке ===")
    prec = precision_at_k(global_model, test_interactions, k=5).mean()
    print(f">>> Итоговая точность Precision@5 на тесте: {prec:.4f}")

    plt.figure(figsize=(5,3))
    plt.bar(["Global model"], [prec], color="#4CAF50")
    plt.ylabel("Precision@5")
    plt.title("Точность глобальной модели (FedAvg)")
    plt.ylim(0,1)
    plt.tight_layout()
    out_dir = os.environ.get("OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.png")
    plt.savefig(out_path)
    print(f"[INFO] График метрики сохранён в {out_path}")

    # Дополнительные графики
    # 1) Глобальная Precision@5 по раундам
    if history and history.get("round_global_precision"):
        plt.figure(figsize=(6,3))
        rounds = list(range(1, len(history["round_global_precision"]) + 1))
        plt.plot(rounds, history["round_global_precision"], marker='o', color='#1976D2')
        plt.xlabel("Раунд")
        plt.ylabel("Precision@5 (тест)")
        plt.title("Глобальная Precision@5 по раундам")
        plt.ylim(0,1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path2 = os.path.join(out_dir, "precision_per_round.png")
        plt.savefig(out_path2)
        print(f"[INFO] Сохранён график динамики качества: {out_path2}")

    # 2) Валидационная Precision@5 по клиентам и раундам (heatmap-like)
    if history and history.get("round_client_precisions"):
        import numpy as np
        client_matrix = np.array([row for row in history["round_client_precisions"]], dtype=object)
        # Преобразуем в числовую матрицу (раунды x клиенты)
        max_clients = max(len(r) for r in history["round_client_precisions"])
        vals = np.zeros((len(history["round_client_precisions"]), max_clients))
        for i, row in enumerate(history["round_client_precisions"]):
            for j, v in enumerate(row):
                vals[i, j] = v
        plt.figure(figsize=(6,3))
        for j in range(vals.shape[1]):
            plt.plot(range(1, vals.shape[0]+1), vals[:, j], marker='o', label=f"Client {j+1}")
        plt.xlabel("Раунд")
        plt.ylabel("Precision@5 (val)")
        plt.title("Валидационная Precision@5 по клиентам")
        plt.ylim(0,1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path3 = os.path.join(out_dir, "client_val_precision_per_round.png")
        plt.savefig(out_path3)
        print(f"[INFO] Сохранён график по клиентам: {out_path3}")

    # 3) Частота выбора arm (эпох) по клиентам
    if history and history.get("round_client_epochs"):
        import collections as _collections
        epochs_per_round = history["round_client_epochs"]
        # Считаем частоты по каждому клиенту
        num_clients = max(len(r) for r in epochs_per_round)
        plt.figure(figsize=(6,3))
        width = 0.25
        all_arms = sorted(set(e for row in epochs_per_round for e in row))
        for c in range(num_clients):
            cnt = _collections.Counter([row[c] for row in epochs_per_round if len(row) > c])
            heights = [cnt.get(a, 0) for a in all_arms]
            plt.bar([x + c*width for x in range(len(all_arms))], heights, width=width, label=f"Client {c+1}")
        plt.xticks([x + width for x in range(len(all_arms))], [str(a) for a in all_arms])
        plt.xlabel("Выбранные эпохи (arm)")
        plt.ylabel("Частота выбора")
        plt.title("Статистика выбора эпох по клиентам")
        plt.legend()
        plt.tight_layout()
        out_path4 = os.path.join(out_dir, "bandit_arms_usage.png")
        plt.savefig(out_path4)
        print(f"[INFO] Сохранён график выбора эпох: {out_path4}")

    user_ids = list(test_df['user_id'].unique())
    random_user = np.random.choice(user_ids)
    user_index = dataset.mapping()[0][random_user]
    scores = global_model.predict(user_index, np.arange(dataset.interactions_shape()[1]))
    top_items = np.argsort(-scores)[:5]
    item_id_map = {v: k for k, v in dataset.mapping()[2].items()}
    top_item_ids = [int(item_id_map[i]) for i in top_items]
    print(f">>> Топ-5 фильмов, которые система рекомендует пользователю {random_user}: {top_item_ids} (ID фильмов MovieLens)")
    print("=== Работа завершена успешно. Все этапы выполнены корректно. ===")
    print("========== [ЗАВЕРШЕНО] ==========") 