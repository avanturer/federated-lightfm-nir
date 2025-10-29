from .data_utils import load_movielens, split_data_among_clients, split_train_test, get_full_user_item_lists, split_client_train_val
from .fedavg import federated_training
from .lightfm_model import build_interaction_matrix_with_dataset
from lightfm.evaluation import precision_at_k
import matplotlib.pyplot as plt
import numpy as np
import os

def _split_clients_time_skew(train_df):
    # Делим train хронологически на доли [60%, 30%, 10%] по времени для усиления неоднородности
    df_sorted = train_df.sort_values('timestamp')
    n = len(df_sorted)
    c1 = df_sorted.iloc[: int(0.6 * n)]
    c2 = df_sorted.iloc[int(0.6 * n): int(0.9 * n)]
    c3 = df_sorted.iloc[int(0.9 * n):]
    return [c1, c2, c3]

def prepare_data(skew_clients: bool = False):
    data = load_movielens()
    train_df, test_df = split_train_test(data, test_size=0.2, random_state=42)
    if skew_clients:
        raw_clients_data = _split_clients_time_skew(train_df)
    else:
        raw_clients_data, _, _ = split_data_among_clients(train_df, n_clients=3, all_data=data)
    clients_data = []
    for cdata in raw_clients_data:
        train_c, val_c = split_client_train_val(cdata, val_size=0.1, random_state=42)
        clients_data.append({'train': train_c, 'val': val_c})
    all_users, all_items = get_full_user_item_lists(data)
    from lightfm.data import Dataset
    dataset = Dataset()
    dataset.fit(all_users, all_items)
    test_interactions = build_interaction_matrix_with_dataset(test_df, dataset)
    return data, clients_data, dataset, test_interactions

def run_config(name, clients_data, dataset, test_interactions, **fed_kwargs):
    model, hist = federated_training(
        clients_data,
        dataset,
        test_interactions=test_interactions,
        **fed_kwargs
    )
    final_prec = float(precision_at_k(model, test_interactions, k=5).mean())
    return {
        'name': name,
        'final_prec': final_prec,
        'round_precisions': hist.get('round_global_precision', [])
    }

if __name__ == "__main__":
    print("[EXP] Подготовка данных...")
    # Базовый (равномерные клиенты)
    _, clients_data, dataset, test_interactions = prepare_data(skew_clients=False)
    out_dir = os.environ.get("OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    common = dict(rounds=8, epoch_arms=[10,20,30], epsilon=0.15, use_weighted_agg=True,
                  no_components=128, loss='warp', learning_schedule='adagrad', learning_rate=0.07,
                  item_alpha=1e-6, user_alpha=1e-6, num_threads_alloc=4, num_threads_fit=4)

    print("[EXP] Запуск конфигураций...")
    results = []

    # 1) Базовая (bandit + weighted, WARP)
    results.append(run_config("bandit_weighted_warp", clients_data, dataset, test_interactions, **common))

    # 2) Без bandit (фиксированные 20 эпох) + weighted
    no_bandit = common.copy()
    no_bandit.update(dict(epoch_arms=[20], epsilon=0.0))
    results.append(run_config("fixed20_weighted_warp", clients_data, dataset, test_interactions, **no_bandit))

    # 3) Unweighted FedAvg (равноправное усреднение)
    unweighted = common.copy()
    unweighted.update(dict(use_weighted_agg=False))
    results.append(run_config("bandit_unweighted_warp", clients_data, dataset, test_interactions, **unweighted))

    # 4) Logistic вместо WARP (остальное как в baseline)
    logistic = common.copy()
    logistic.update(dict(loss='logistic'))
    results.append(run_config("bandit_weighted_logistic", clients_data, dataset, test_interactions, **logistic))

    # Сценарий с неоднородными клиентами (усиливаем разницу, чтобы показать пользу bandit)
    print("[EXP] Подготовка данных (клиенты со смещением по времени)...")
    _, clients_skew, dataset_skew, test_inter_skew = prepare_data(skew_clients=True)
    skew_cfg = dict(rounds=10, epoch_arms=[5,15,30,60], epsilon=0.2, use_weighted_agg=True,
                    no_components=128, loss='warp', learning_schedule='adagrad', learning_rate=0.07,
                    item_alpha=1e-6, user_alpha=1e-6, num_threads_alloc=4, num_threads_fit=4)
    results.append(run_config("SKew_bandit_weighted_warp", clients_skew, dataset_skew, test_inter_skew, **skew_cfg))
    fixed_skew = skew_cfg.copy(); fixed_skew.update(dict(epoch_arms=[20], epsilon=0.0))
    results.append(run_config("SKew_fixed20_weighted_warp", clients_skew, dataset_skew, test_inter_skew, **fixed_skew))

    # Сохраняем CSV
    csv_path = os.path.join(out_dir, "ablation_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("name,final_precision\n")
        for r in results:
            f.write(f"{r['name']},{r['final_prec']:.6f}\n")
    print(f"[EXP] Результаты сохранены: {csv_path}")

    # Барчарт финальных метрик
    plt.figure(figsize=(7,3))
    names = [r['name'] for r in results]
    vals = [r['final_prec'] for r in results]
    plt.bar(names, vals, color=["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"])
    plt.xticks(rotation=20, ha='right')
    plt.ylabel("Precision@5 (test)")
    plt.title("Ablation: final Precision@5")
    plt.tight_layout()
    out_bar = os.path.join(out_dir, "ablation_bars.png")
    plt.savefig(out_bar)
    print(f"[EXP] Сохранён график: {out_bar}")

    # Линии по раундам (если есть)
    max_rounds = max(len(r['round_precisions']) for r in results)
    plt.figure(figsize=(7,3))
    for r, c in zip(results, ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]):
        if r['round_precisions'] and any(v is not None for v in r['round_precisions']):
            y = [v if v is not None else np.nan for v in r['round_precisions']]
            x = list(range(1, len(y)+1))
            plt.plot(x, y, marker='o', label=r['name'], color=c)
    plt.xlabel("Раунд")
    plt.ylabel("Precision@5 (test)")
    plt.title("Ablation: per-round global Precision@5")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_line = os.path.join(out_dir, "ablation_rounds.png")
    plt.savefig(out_line)
    print(f"[EXP] Сохранён график: {out_line}")

