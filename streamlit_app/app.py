import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon=""
)

st.title("EMIPredict AI")
st.write("Application is running successfully!")

st.success("App is working!")

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Predict"])

if page == "Home":
    st.write("### Welcome")
    col1, col2, col3 = st.columns(3)
    col1.metric("Models", "6")
    col2.metric("Data", "400,000")
    col3.metric("Accuracy", "99%")

elif page == "Predict":
    st.write("### Prediction")
    age = st.slider("Age", 18, 65, 30)
    income = st.slider("Income", 10000, 200000, 50000)
    
    if st.button("Predict"):
        st.success("Eligible!")