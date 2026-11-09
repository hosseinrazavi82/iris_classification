
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸"
)


# Train Machine Learning Model
@st.cache_resource
def train_model():

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200)
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, iris, accuracy


model, iris, accuracy = train_model()


# Web App
st.title("🌸 Iris Flower Classification")

st.write(
    "Enter flower measurements to predict its species."
)

st.metric("Model Accuracy", f"{accuracy:.2%}")

st.divider()

st.subheader("Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)", 0.0, 10.0, 5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)", 0.0, 10.0, 3.5
)

petal_length = st.number_input(
    "Petal Length (cm)", 0.0, 10.0, 1.4
)

petal_width = st.number_input(
    "Petal Width (cm)", 0.0, 10.0, 0.2
)


if st.button("Predict Flower 🌸"):

    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]

    st.success(
        f"Predicted Species: "
        f"{iris.target_names[prediction].title()}"
    )

    st.subheader("Prediction Probabilities")

    for name, probability in zip(
        iris.target_names, probabilities
    ):
        st.write(f"{name.title()}: {probability:.2%}")
        st.progress(float(probability))
        