# ML FastAPI сервис с Docker

Учебный проект: REST API для инференса ML-модели на базе **FastAPI**, с тестами, линтингом и контейнеризацией через **Docker**.

Реализована модель классификации токсичности русского текста  
`s-nlp/russian_toxicity_classifier` (Hugging Face).

---

## Назначение проекта

Проект демонстрирует полный цикл разработки ML-сервиса:

- загрузка и кэширование модели
- REST API для инференса
- валидация входных данных
- автоматические тесты
- контроль стиля кода
- сборка и запуск в Docker

Проект выполнен в рамках лабораторной работы.

---

## Стек технологий

- Python 3.11  
- FastAPI  
- Uvicorn  
- Hugging Face Transformers  
- PyTorch  
- Pytest  
- Docker  
- pre-commit  
- black / flake8 / isort  

---

## Архитектура проекта

```
ml_app/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI приложение и маршруты
│   ├── model.py       # Загрузка модели и инференс
│   └── schemas.py     # Pydantic-схемы
├── tests/
│   └── test_api.py    # API-тесты
├── Dockerfile
├── requirements.txt
├── .pre-commit-config.yaml
├── .gitignore
└── README.md
```

Модель загружается один раз и кэшируется в памяти процесса.

---

## API эндпоинты

### Проверка состояния
```
GET /health
```

Ответ:
```json
{
  "status": "ok"
}
```

---

### Инференс одного текста
```
POST /predict
```

Тело запроса:
```json
{
  "text": "Пример текста"
}
```

Ответ:
```json
{
  "text": "Пример текста",
  "prediction": {
    "label": "non-toxic",
    "score": 0.95
  }
}
```

---

### Пакетный инференс
```
POST /predict_batch
```

Тело запроса:
```json
{
  "texts": [
    "Первый текст",
    "Второй текст"
  ]
}
```

Ответ:
```json
{
  "results": [
    {
      "text": "Первый текст",
      "prediction": {
        "label": "non-toxic",
        "score": 0.98
      }
    },
    {
      "text": "Второй текст",
      "prediction": {
        "label": "toxic",
        "score": 0.87
      }
    }
  ]
}
```

---

### Информация о модели
```
GET /model_info
```

---

### Swagger UI
```
GET /docs
```

---

## Локальный запуск (без Docker)

1. Создать виртуальное окружение:
```
python -m venv venv
```

2. Активировать:
```
venv\Scripts\activate      # Windows
source venv/bin/activate  # Linux / macOS
```

3. Установить зависимости:
```
pip install -r requirements.txt
```

4. Запустить сервер:
```
uvicorn app.main:app --reload
```

Адрес:
```
http://localhost:8000
```

---

## Запуск тестов

```
pytest -q
```

Ожидаемый результат:
```
5 passed
```

---

## Docker

### Сборка образа
```
docker build -t ml_app:latest .
```

### Запуск контейнера
```
docker run --rm -p 8000:8000 ml_app:latest
```

---

## Стиль кода и pre-commit

Используются:
- black
- flake8
- isort

Установка хуков:
```
pre-commit install
```

Ручная проверка:
```
pre-commit run --all-files
```

Коммиты блокируются при нарушении стиля.

---

## Проверка корректности работы

Чеклист:

- `pytest` проходит без ошибок  
- `/health` возвращает HTTP 200  
- `/predict` и `/predict_batch` возвращают валидный JSON  
- Docker-образ собирается и запускается  
- В репозитории отсутствуют:
  - `venv/`
  - `.cache/`
  - веса модели  

---

## Примечания

- Модель загружается с Hugging Face при первом запросе.
- Требуется интернет при первом запуске.
- Веса модели не коммитятся в репозиторий.
- Проект ориентирован на учебные цели.
