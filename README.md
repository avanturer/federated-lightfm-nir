# Federated LightFM Recommender (НИР)

[![GitHub repo](https://img.shields.io/badge/GitHub-avanturer%2Ffederated--lightfm--nir-blue?logo=github)](https://github.com/avanturer/federated-lightfm-nir)

## Описание

Данный проект посвящён исследованию федеративного обучения рекомендательных систем на примере датасета MovieLens и модели LightFM. В работе реализован прототип, в котором для каждого клиента динамически выбирается число эпох обучения с помощью epsilon-greedy bandit. Проект выполнен в рамках научно-исследовательской работы (НИР) на кафедре инженерной кибернетики НИТУ МИСИС.

**Автор:** Оркин Родион Родионович, 3 курс, направление "Прикладная математика"  
**Научный руководитель:** Курочкин Илья Ильич

---

## Требования
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, Mac, Linux)
- Доступ к интернету для скачивания датасета MovieLens

---

## Установка и запуск

1. **Клонируйте репозиторий:**
   ```sh
   git clone https://github.com/avanturer/federated-lightfm-nir.git
   cd federated-lightfm-nir
   ```
2. **Убедитесь, что Docker Desktop запущен.**
3. **Соберите Docker-образ:**
   ```sh
   docker build -t fed-lightfm .
   ```
4. **Запустите обучение:**
   - **Windows (PowerShell):**
     ```sh
     docker run --rm -v ${PWD}:/app fed-lightfm
     ```
   - **Linux/Mac:**
     ```sh
     docker run --rm -v $(pwd):/app fed-lightfm
     ```
   После выполнения появится файл `result.png` и подробный вывод в консоль.

> **Примечание:** Если файл `result.png` не появился в папке проекта, проверьте, что Docker Desktop запущен и volume смонтирован правильно.

---

## Структура проекта

```
├── Dockerfile
├── requirements.txt
├── README.md
├── result.png                # График итоговой метрики
└── src/
    ├── main.py               # Точка входа
    ├── fedavg.py             # Логика FedAvg и bandit
    ├── lightfm_model.py      # Модель LightFM
    ├── data_utils.py         # Работа с данными
    └── bandit.py             # EpsilonGreedyBandit
```

---

## Описание работы

- Данные MovieLens автоматически скачиваются и делятся между 3 клиентами.
- Каждый клиент обучает LightFM на своей части данных, число эпох выбирается с помощью bandit-алгоритма.
- После каждого раунда FedAvg происходит усреднение весов моделей.
- В консоли отображаются этапы обучения, Precision@5 на валидации, статистика по arms bandit'а, итоговая метрика Precision@5 на тесте, топ-5 фильмов для случайного пользователя.
- Итоговый график Precision@5 сохраняется в `result.png`.

---

## Пример вывода

```
=== FedAvg: Раунд 1/3 (обучение локальных моделей и усреднение) ===
  [CLIENT 1] Выбранное число эпох: 3
    Precision@5 на валидации: 0.0154
  ...
=== Статистика выбора числа эпох (bandit) по клиентам ===
[CLIENT 1] Arms usage:
  Эпох: 3 | Выбран: 3 раз | Средний Precision@5: 0.0152
  ...
=== Оценка итоговой глобальной модели на тестовой выборке ===
>>> Итоговая точность Precision@5 на тесте: 0.1121
[INFO] График метрики сохранён в result.png
>>> Топ-5 фильмов, которые система рекомендует пользователю 515: [50, 258, 181, 288, 286] (ID фильмов MovieLens)
=== Работа завершена успешно. Все этапы выполнены корректно. ===
```

---

## Метрики
- **Precision@5:** средняя точность топ-5 рекомендаций.
- **Статистика bandit:** количество выборов каждого arm (число эпох), средний Precision@5 для каждого arm.

---

## Ссылки
- [Репозиторий на GitHub](https://github.com/avanturer/federated-lightfm-nir)
- [LightFM на GitHub](https://github.com/lyst/lightfm)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

---

## Контакты
- Оркин Родион Родионович, НИТУ МИСИС, 3 курс, кафедра инженерной кибернетики
- Научный руководитель: Курочкин Илья Ильич 