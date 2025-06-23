from .data_utils import load_movielens, split_data_among_clients, split_train_test, get_full_user_item_lists, split_client_train_val
from .fedavg import federated_training, get_model_weights
from .lightfm_model import build_interaction_matrix, build_interaction_matrix_with_dataset, train_lightfm
import numpy as np
import matplotlib.pyplot as plt
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
    global_model = federated_training(clients_data, dataset, rounds=3, epoch_arms=[3, 5, 10], epsilon=0.2)
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
    plt.savefig("result.png")
    print("[INFO] График метрики сохранён в result.png")

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