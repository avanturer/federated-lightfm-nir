# Federated LightFM Recommender (НИР)

[![GitHub repo](https://img.shields.io/badge/GitHub-avanturer%2Ffederated--lightfm--nir-blue?logo=github)](https://github.com/avanturer/federated-lightfm-nir)

## Описание

Проект посвящён динамическому федеративному обучению для рекомендаций. Реализованы два контура:

- Локальный FedAvg с моделью `LightFM` и динамическим выбором числа эпох через epsilon-greedy bandit (прототип на одном узле). Применяется WARP‑обучение, хронологические разбиения без «подглядывания» и логирование метрик по раундам.
- Распределённый контур на `Flower (FL)` с глубокой моделью `NeuralCF (PyTorch)` и дифференциальной приватностью (клиппинг + гауссовский шум на обновлениях). Для анализа потоков метрик добавлен минимальный `PyFlink` job (локальный runner).

Проект выполнен в рамках НИР; акцент на воспроизводимости и минимальной рабочей конфигурации.

**Автор:** Оркин Родион Родионович, 3 курс, направление "Прикладная математика"  
**Научный руководитель:** Курочкин Илья Ильич

---

## Требования
- Docker (или локальная среда Python 3.10)
- Интернет для скачивания MovieLens 100k

---

## Установка и запуск

1. **Клонируйте репозиторий:**
   ```sh
   git clone https://github.com/avanturer/federated-lightfm-nir.git
   cd federated-lightfm-nir
   ```
2. **Убедитесь, что Docker Desktop запущен.**
3. **Соберите Docker-образ (один раз):**
   ```sh
   docker build -t fed-lightfm .
   ```
4. **Запустите локальный прототип (LightFM + FedAvg + bandit):**
   - **Windows (PowerShell):**
     ```sh
     docker run --rm -v ${PWD}:/app fed-lightfm
     ```
   - **Linux/Mac:**
     ```sh
     docker run --rm -v $(pwd):/app fed-lightfm
     ```
   После выполнения появятся графики в `out/` и подробный вывод в консоль.

5. (Опционально) **Запуск распределённого FL (Flower, PyTorch, DP):**
   - В одном терминале запустите сервер:
     ```sh
     python -m federated_flower.server
     ```
   - В трёх других терминалах запустите 3 клиента:
     ```sh
     CLIENT_ID=0 python -m federated_flower.client
     CLIENT_ID=1 python -m federated_flower.client
     CLIENT_ID=2 python -m federated_flower.client
     ```
   Сервер запускает стратегию FedAvg с bandit-конфигом для числа эпох и опциями DP (клиппинг и шум) через fit-config.

6. (Опционально) **Анализ метрик через PyFlink (локальный runner):**
   Подайте CSV вида `timestamp,reward` в джоб:
   ```sh
   python -m federated_flower.flink_job rewards.csv rewards_with_mavg.txt
   ```

> **Примечание:** Если файл `result.png` не появился в папке проекта, проверьте, что Docker Desktop запущен и volume смонтирован правильно.

---

## Структура проекта

```
├── Dockerfile
├── requirements.txt
├── README.md
├── result.png                # График итоговой метрики
├── federated_flower/
│   ├── server.py             # Flower сервер со стратегией и bandit
│   ├── client.py             # Flower клиент с NeuralCF (PyTorch) + DP
│   ├── ncf_model.py          # Модель Neural Collaborative Filtering
│   ├── data.py               # Загрузка/маппинг данных, DataLoader
│   └── flink_job.py          # Минимальный PyFlink job (moving average)
└── src/
    ├── main.py               # Точка входа
    ├── fedavg.py             # Логика FedAvg и bandit
    ├── lightfm_model.py      # Модель LightFM
    ├── data_utils.py         # Работа с данными
    └── bandit.py             # EpsilonGreedyBandit
```

---

## Описание работы (локальный контур)

- Загружаются MovieLens 100k, затем выполняется хронологический сплит: ранние записи → train, поздние → test (без утечек).
- Train делится между 3 клиентами; внутри каждого клиента валидируется по времени (последние 10%).
- В каждом раунде FedAvg клиенты обучают `LightFM` (WARP, 64–128 компонент, Adagrad, слабая регуляризация) локально. Число эпох выбирается из набора arms [5, 10, 15, 20] с помощью epsilon‑greedy; epsilon плавно уменьшается по раундам.
- После раунда выполняется FedAvg‑усреднение весов. Логируются: валидационные метрики по клиентам, глобальная Precision@5 по раундам, статистика выбора arms.
- Сохраняются графики: `result.png` (итог), `precision_per_round.png`, `client_val_precision_per_round.png`, `bandit_arms_usage.png`.

### Абляционные сравнения

Для воспроизведения сравнения (bandit vs fixed, weighted vs unweighted, WARP vs logistic):
```sh
docker run --rm -e OUTPUT_DIR=/app/out -v $(pwd):/app fed-lightfm micromamba run -n base python -m src.experiments
```
Будут сохранены: `ablation_results.csv`, `ablation_bars.png`, `ablation_rounds.png` в папке `out/`.

В распределённом контуре (Flower):
- Сервер рассылает клиентам конфиг с выбранным числом эпох (bandit) и параметрами DP.
- Клиенты обучают `NeuralCF` и возвращают обновления параметров, на которые применяются клиппинг и шум (дифференциальная приватность) на стороне клиента.
- Агрегация выполняется FedAvg. Для примера arms алгоритмов сейчас задействован `ncf` (расширение до нескольких архитектур возможно без изменения протокола).

---

## Пример вывода (фрагмент)

```
=== FedAvg: Раунд 1/7 ===
  [CLIENT 1] Выбранное число эпох: 5
    Precision@5 на валидации: 0.0959
  ...
=== Статистика выбора числа эпох (bandit) по клиентам ===
[CLIENT 1] Arms usage:
  Эпох: 5 | Выбран: 7 раз | Средний Precision@5: 0.0959
  Эпох: 10 | Выбран: 0 раз | Средний Precision@5: N/A
  ...
=== Оценка итоговой глобальной модели на тестовой выборке ===
>>> Итоговая точность Precision@5 на тесте: ~0.30 (варьируется от прогона)
[INFO] Графики сохранены в папке out/
```

---

## Метрики
- **Precision@5:** средняя точность топ-5 рекомендаций.
- **Статистика bandit:** количество выборов каждого arm (число эпох), средний Precision@5 для каждого arm (если arm не выбирался, выводится N/A).
 - **Loss (Flower):** средний BCE-loss по валидации (снижение соответствует росту качества).

---

## Ссылки
- [Репозиторий на GitHub](https://github.com/avanturer/federated-lightfm-nir)
- [LightFM на GitHub](https://github.com/lyst/lightfm)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

---

## Контакты
- Оркин Родион Родионович, НИТУ МИСИС, 4 курс, кафедра инженерной кибернетики
- Научный руководитель: Курочкин Илья Ильич 