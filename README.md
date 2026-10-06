# Recommender System Library

Учебная Python-библиотека с набором алгоритмов рекомендательных систем, метрик качества и вспомогательных функций для работы с разреженными матрицами user-item. В репозитории также находятся Flask API для хранения и асинхронного обучения моделей, примеры запросов, Jupyter-ноутбуки и результаты экспериментов.

## Возможности

- memory-based рекомендации по сходству пользователей и объектов;
- матричная факторизация для explicit feedback;
- модели для implicit feedback;
- обучение и дообучение моделей на `scipy.sparse.coo_matrix`;
- прогноз оценок и получение ранжированного списка объектов;
- метрики качества рекомендаций;
- чтение CSV, построение и сохранение разреженных матриц;
- сериализация моделей;
- локальный REST API с токенами пользователей и SQLite-хранилищем.

## Реализованные модели

| Группа | Класс | Основные параметры |
| --- | --- | --- |
| Memory-based | `UserBasedModel` | `k_nearest_neighbours` |
| Memory-based | `ItemBasedModel` | `k_nearest_neighbours`, `barrier_type` (`mean` или `median`) |
| Latent factor | `SingularValueDecompositionModel` | `dimension` |
| Latent factor | `AlternatingLeastSquaresModel` | `dimension` |
| Latent factor | `HierarchicalAlternatingLeastSquaresModel` | `dimension` |
| Latent factor | `StochasticLatentFactorModel` | `dimension`, `learning_rate`, регуляризация пользователей и объектов |
| Implicit | `ImplicitAlternatingLeastSquaresModel` | `dimension`, `influence_regularization` |
| Implicit | `ImplicitHierarchicalAlternatingLeastSquaresModel` | `dimension`, `influence_regularization` |
| Implicit | `ImplicitStochasticLatentFactorModel` | `dimension`, `learning_rate`, коэффициенты регуляризации |

Метрики:

- Precision@k и Recall@k;
- F1;
- ROC AUC;
- NDCG;
- MAE, MSE и RMSE.

## Структура репозитория

```text
recommender_system_library/   устанавливаемая библиотека recommender_systems
recommender_system_api/       Flask API, SQLite и сохранённые модели
experiments/                  скрипты, ноутбуки, тестовые данные и графики
```

Основные модули библиотеки:

- `recommender_systems.models` — модели и общие абстракции;
- `recommender_systems.metrics` — функции оценки качества;
- `recommender_systems.extra_functions` — работа с CSV, матрицами, рейтингами и сериализацией.

## Требования

Проект использует версии зависимостей 2021 года. Для воспроизводимого запуска рекомендуется Python 3.8 или 3.9 и зависимости из [`recommender_system_library/setup.py`](./recommender_system_library/setup.py).

> Код использует устаревшие псевдонимы `numpy.int` и `numpy.float`, удалённые в новых версиях NumPy. Оставьте зафиксированную версию `numpy==1.20.3` либо обновите эти обращения перед переходом на современный NumPy.

## Установка библиотеки

Клонируйте репозиторий по SSH и создайте виртуальное окружение:

```bash
git clone git@github.com:nov-stacy/Recommender-System-Library.git
cd Recommender-System-Library

python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./recommender_system_library
```

В PowerShell активация окружения выполняется так:

```powershell
.venv\Scripts\Activate.ps1
```

## Быстрый пример

```python
import numpy as np
from scipy import sparse

from recommender_systems.models.latent_factor_models import (
    SingularValueDecompositionModel,
)


ratings = sparse.coo_matrix(
    np.array(
        [
            [5.0, 4.0, 0.0, 1.0],
            [4.0, 0.0, 3.0, 1.0],
            [1.0, 1.0, 0.0, 5.0],
        ]
    )
)

model = SingularValueDecompositionModel(dimension=2).fit(ratings)

# Оценки всех объектов для пользователя с индексом 0
predicted_ratings = model.predict_ratings(0)

# Индексы всех объектов в порядке убывания прогнозируемой оценки
ranked_items = model.predict(0)

print(predicted_ratings)
print(ranked_items)
```

`predict()` возвращает полный рейтинг объектов и не исключает уже просмотренные элементы — при необходимости отфильтруйте их на стороне приложения.

### Обучение итеративной модели

```python
from recommender_systems.models.latent_factor_models import (
    StochasticLatentFactorModel,
)


model = StochasticLatentFactorModel(
    dimension=8,
    learning_rate=0.001,
    user_regularization=0.1,
    item_regularization=0.1,
)

model.fit(ratings, epochs=20, debug_name="rmse", verbose=True)
print(model.debug_information.get())
```

Для `debug_name` поддерживаются `mse`, `mae`, `rmse` или `None`.

## Подготовка данных

Модели принимают строго `scipy.sparse.coo_matrix`, где строки соответствуют пользователям, столбцы — объектам, а значения — оценкам или сигналам взаимодействия.

```python
from recommender_systems.extra_functions.work_with_tables import (
    generate_sparse_matrix,
    read_data_from_csv,
)


table = read_data_from_csv("ratings.csv")
ratings = generate_sparse_matrix(
    table,
    column_user_id="user_id",
    column_item_id="item_id",
    column_rating="rating",
)
```

`generate_sparse_matrix()` преобразует уникальные ID пользователей и объектов во внутренние последовательные индексы, но не возвращает таблицы соответствия. Если исходные ID нужны после прогнозирования, сохраните такое соответствие отдельно.

Дополнительные функции позволяют:

- разделить известные оценки с помощью `get_train_matrix()`;
- сохранить/загрузить матрицу через `write_matrix_to_file()` и `read_matrix_from_file()`;
- сохранить/загрузить модель через `save_model_to_file()` и `get_model_from_file()`;
- получить top-k либо рекомендации выше порога через `calculate_predicted_items()`.

## Запуск REST API

Сначала установите библиотеку локально, затем зависимости Flask:

```bash
python -m pip install -e ./recommender_system_library
python -m pip install \
  Flask==2.0.0 Werkzeug==2.0.1 Jinja2==3.0.1 \
  itsdangerous==2.0.1 click==8.0.1

cd recommender_system_api
python main.py
```

Сервер запускается на `http://127.0.0.1:5000`. Запускайте его именно из каталога `recommender_system_api`, поскольку пути к `database/` заданы относительно текущего рабочего каталога.

Файл [`recommender_system_api/requirements.txt`](./recommender_system_api/requirements.txt) отражает исходное окружение, но содержит платформозависимую строку `pkg-resources==0.0.0` и устанавливает библиотеку из старого Git-коммита. Поэтому для работы с текущим checkout рекомендуется установка командами выше.

### Основные маршруты

| Метод | Маршрут | Назначение |
| --- | --- | --- |
| `POST` | `/registration` | создать пользователя и получить токен |
| `POST` | `/create` | создать модель |
| `POST` | `/change/<system_id>` | изменить параметры модели |
| `POST` | `/train/<system_id>` | запустить обучение в фоновом потоке |
| `POST` | `/status/<system_id>` | получить статус обучения |
| `POST` | `/clear/<system_id>` | сбросить модель в необученное состояние |
| `DELETE` | `/delete/<system_id>` | удалить модель |
| `GET` | `/predict_ratings/<system_id>` | получить прогноз оценок |
| `GET` | `/predict_items/<system_id>` | получить ранжированный список объектов |

После регистрации передавайте полученный токен в HTTP-заголовке `token`:

```bash
curl -X POST http://127.0.0.1:5000/registration
```

```bash
curl -X POST http://127.0.0.1:5000/create \
  -H "Content-Type: application/json" \
  -H "token: <TOKEN>" \
  -d '{
    "type": "latent_factor_svd_model",
    "params": {"dimension": 8}
  }'
```

Допустимые значения `type` перечислены в `MODELS_NAMES` внутри [`work_with_models.py`](./recommender_system_library/recommender_systems/extra_functions/work_with_models.py). Полные примеры всех запросов находятся в [`experiments/experiments_with_api`](./experiments/experiments_with_api).

### Формат данных обучения API

`POST /train/<system_id>` ожидает JSON с параметрами обучения и сериализованной матрицей:

```json
{
  "params": {"epochs": 20},
  "train_data": "<base64 от pickle с scipy.sparse.coo_matrix>"
}
```

Статус принимает значения `TRAINING`, `READY` или `ERROR DURING THE TRAINING`.

> API десериализует `pickle` из запроса и запускается с `debug=True`. Не публикуйте этот сервер в интернете и не принимайте данные от недоверенных клиентов: вредоносный pickle способен выполнить произвольный код. API предназначен только для локальных экспериментов.

## Тесты и эксперименты

В библиотеке находится 17 файлов с unit-тестами. После установки зависимостей их можно запустить командой:

```bash
python -m unittest discover \
  -s recommender_system_library/recommender_systems \
  -p "test_*.py"
```

Каталог [`experiments`](./experiments) содержит:

- скрипты проверки параметров алгоритмов;
- примеры работы с библиотекой и API;
- Jupyter-ноутбуки;
- исходные таблицы и разреженные матрицы;
- сохранённые графики Precision@k, F1, AUC, NDCG и ошибок.

## Ограничения текущей версии

- версии зависимостей устарели и не совместимы с современным NumPy без изменений кода;
- автоматический CI отсутствует;
- библиотека имеет версию `0` и не опубликована как стабильный пакет;
- API хранит модели в `pickle`, состояние — в локальной SQLite, а задачи обучения — только в памяти процесса;
- HTTP API использует GET-запросы с JSON-телом для прогнозов, что поддерживается не всеми клиентами и прокси;
- некоторые демонстрационные данные и обученные модели уже сохранены в репозитории;
- `setup.py` объявляет лицензию MIT, но отдельный файл `LICENSE` в репозитории отсутствует.
