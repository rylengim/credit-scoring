# Credit Scoring App

![Форма заявки](misc/form.png)
Сервис для оценки риска дефолта по заявке на кредитную карту.

## Стек
- Python
- scikit-learn
- Streamlit
- Docker

## Запуск локально
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Запуск через Docker
```bash
docker compose up --build
```

После запуска приложение будет доступно по адресу:
[http://localhost:8501](http://localhost:8501)

## Структура проекта
- `prepare.py` — подготовка признаков
- `train.py` — обучение модели
- `app.py` — Streamlit-интерфейс
- `model.pkl` — обученная модель
- `data/scoring_features.csv` — подготовленные данные

