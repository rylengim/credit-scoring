from pathlib import Path

import streamlit as st
import joblib
import pandas as pd

from features import prepare_features


st.set_page_config(page_title="Кредитный скоринг — прототип", page_icon="💳")

st.title("Кредитный скоринг — прототип")
st.write("Учебная демонстрация оценки риска на данных репозитория")


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).resolve().with_name("model.pkl"))


model = load_model()

st.sidebar.header("Параметры скоринга")
threshold = st.sidebar.slider(
    "Порог отказа по риску дефолта",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)
st.sidebar.write(f"Текущий порог: {threshold:.0%}")

with st.form("Подать заявку"):
    age = st.number_input("Ваш возраст", min_value=18, max_value=100, step=1)
    income = st.number_input("Ваш доход в тысячах рублей", min_value=0.0, step=1.0)
    education = st.checkbox("У меня есть высшее образование")
    work = st.checkbox("У меня есть стабильная работа")
    car = st.checkbox("У меня есть автомобиль")
    submit = st.form_submit_button("Подать заявку")


if submit:
    features = prepare_features(pd.DataFrame([{
        "age": age, "income": income, "education": education,
        "work": work, "car": car,
    }]))

    default_proba = model.predict_proba(features.to_numpy())[0][1]

    st.subheader("Демонстрационный результат")
    st.write(f"Вероятность дефолта: **{default_proba:.1%}**")
    st.write(f"Порог отказа: **{threshold:.0%}**")

    if default_proba >= threshold:
        st.error("Оценка риска выше порога: условный отказ.")
    else:
        st.success("Оценка риска ниже порога: условное одобрение.")

    with st.expander("Показать рассчитанные признаки"):
        st.json(features.iloc[0].round(3).to_dict())
