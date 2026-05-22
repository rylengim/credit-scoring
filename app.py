# import streamlit as st
# import requests

# st.title("Кредитная карта Premium")
# st.write("Новая кредитная карта с мгновенным одобрением")

# with st.form("Подать заявку"):
#     age = st.number_input("Ваш возраст", min_value=18)
#     income = st.number_input("Ваш доход в тысячах рублей", min_value=0)
#     education = st.checkbox("У меня есть высшее образование")
#     work = st.checkbox("У меня есть стабильная работа")
#     car = st.checkbox("У меня есть автомобиль")
#     submit = st.form_submit_button('Подать заявку')

# if submit:
#     data = {
#         "age": age,
#         "income": income,
#         "education": education,
#         "work": work,
#         "car": car,
#     }
#     response = requests.post("http://127.0.0.1:8000/score", json=data)
#     if response.json()["approved"]:
#         st.success("Поздравляем, ваша заявка одобрена!")
#     else:
#         st.success("Подобрали для Вас дебетовую карту с 3% кэшбеком.")


# app.py
import streamlit as st
import joblib
import numpy as np


st.set_page_config(page_title="Кредитная карта Premium", page_icon="💳")

st.title("Кредитная карта Premium")
st.write("Новая кредитная карта с мгновенным решением по заявке")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


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
    education = int(education)
    work = int(work)
    car = int(car)

    log_income = np.log1p(income)
    age_sq = age ** 2
    income_per_age = income / age if age != 0 else 0
    employed_educated = work * education

    features = np.array([[
        log_income,
        age,
        age_sq,
        income_per_age,
        employed_educated,
        education,
        work,
        car
    ]])

    default_proba = model.predict_proba(features)[0][1]

    st.subheader("Результат скоринга")
    st.write(f"Вероятность дефолта: **{default_proba:.1%}**")
    st.write(f"Порог отказа: **{threshold:.0%}**")

    if default_proba >= threshold:
        st.error("Высокий риск дефолта. В заявке отказано.")
        st.info("Вместо этого можем предложить дебетовую карту с кэшбэком 3%.")
    else:
        st.success("Поздравляем, ваша заявка одобрена!")

    with st.expander("Показать рассчитанные признаки"):
        st.json({
            "log_income": round(float(log_income), 3),
            "age": int(age),
            "age_sq": int(age_sq),
            "income_per_age": round(float(income_per_age), 3),
            "employed_educated": int(employed_educated),
            "education": int(education),
            "work": int(work),
            "car": int(car),
        })