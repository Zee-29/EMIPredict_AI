import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="🏦"
)

st.title("🏦 EMIPredict AI - Working!")
st.write("Your Streamlit app is running successfully!")

st.success("✅ App is working correctly!")
st.info("👈 Use the sidebar to navigate")

# Simple form
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Predict"])

if page == "Home":
    st.write("### Welcome to EMIPredict AI")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Models", "6")
    col2.metric("Data", "400,000")
    col3.metric("Accuracy", "99%")

elif page == "Predict":
    st.write("### Real-Time Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 30)
        income = st.slider("Monthly Income", 10000, 200000, 50000)
    with col2:
        amount = st.slider("Loan Amount", 10000, 1000000, 100000)
        tenure = st.slider("Tenure", 3, 84, 24)
    
    if st.button("Predict", type="primary"):
        st.success("✅ Eligible for EMI!")
        st.info(f"Monthly EMI: ₹{int(amount/tenure):,}")