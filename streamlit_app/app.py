import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    page_icon="🏦",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists('models/light_classifier.pkl'):
            models['classifier'] = joblib.load('models/light_classifier.pkl')
        else:
            models['classifier'] = None
    except:
        models['classifier'] = None
    
    try:
        if os.path.exists('models/light_regressor.pkl'):
            models['regressor'] = joblib.load('models/light_regressor.pkl')
        else:
            models['regressor'] = None
    except:
        models['regressor'] = None
    
    return models

models = load_models()

# Navigation
with st.sidebar:
    st.title("🏦 EMIPredict AI")
    st.markdown("---")
    page = st.radio("📋 Navigation", ["🏠 Home", "🔮 Predict"])

# Home Page
if page == "🏠 Home":
    st.title("🏦 EMIPredict AI")
    st.subheader("Intelligent Financial Risk Assessment Platform")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Models", "6", "3 Classification + 3 Regression")
    with col2:
        st.metric("📁 Data", "400,000", "Records")
    with col3:
        st.metric("🎯 Accuracy", "99%", "Classification")
    with col4:
        st.metric("📉 RMSE", "1,850", "Regression")
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 What This App Does
    - **EMI Eligibility Classification**: Predicts if you're Eligible, High_Risk, or Not_Eligible
    - **Max Monthly EMI Prediction**: Calculates your maximum safe EMI amount
    - **Real-time Risk Assessment**: Instant analysis based on your financial profile
    """)
    
    if models['classifier'] is None or models['regressor'] is None:
        st.warning("⚠️ Models not loaded. Please train models first or check model files.")

# Predict Page
elif page == "🔮 Predict":
    st.title("🔮 Real-Time Prediction")
    st.markdown("Enter your financial details for instant risk assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        age = st.slider("Age", 18, 65, 30, help="Your age in years")
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        
        st.subheader("💼 Employment Details")
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_exp = st.slider("Years of Experience", 0, 35, 5)
        company_type = st.selectbox("Company Type", ["Startup", "SME", "Large Corporate", "MNC"])
    
    with col2:
        st.subheader("💰 Financial Information")
        monthly_income = st.number_input("Monthly Income (INR)", min_value=10000, max_value=200000, value=50000, step=5000)
        credit_score = st.slider("Credit Score", 300, 850, 700, help="300-850 scale")
        existing_loans = st.selectbox("Existing Loans", ["None", "1", "2", "3+"])
        current_emi = st.number_input("Current EMI (INR)", min_value=0, max_value=50000, value=5000, step=1000)
        
        st.subheader("🏦 Loan Details")
        emi_scenario = st.selectbox("EMI Scenario", 
            ["E-commerce Shopping", "Home Appliances", "Vehicle", "Personal Loan", "Education"])
        requested_amount = st.number_input("Requested Loan Amount (INR)", min_value=10000, max_value=1500000, value=100000, step=10000)
        requested_tenure = st.slider("Requested Tenure (months)", 3, 84, 24)
        
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0, max_value=1000000, value=100000, step=10000)
        emergency_fund = st.number_input("Emergency Fund (INR)", min_value=0, max_value=500000, value=50000, step=5000)
    
    # Predict button
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [1 if gender == "Male" else 0],
            'marital_status': [["Single", "Married", "Divorced"].index(marital_status)],
            'education': [["High School", "Graduate", "Post Graduate", "Professional"].index(education)],
            'employment_type': [["Private", "Government", "Self-employed"].index(employment_type)],
            'years_of_employment': [years_exp],
            'company_type': [["Startup", "SME", "Large Corporate", "MNC"].index(company_type)],
            'house_type': [0],
            'monthly_rent': [0],
            'family_size': [2],
            'dependents': [1],
            'school_fees': [0],
            'college_fees': [0],
            'travel_expenses': [5000],
            'groceries_utilities': [15000],
            'other_monthly_expenses': [5000],
            'existing_loans': [["None", "1", "2", "3+"].index(existing_loans)],
            'current_emi_amount': [current_emi],
            'credit_score': [credit_score],
            'bank_balance': [bank_balance],
            'emergency_fund': [emergency_fund],
            'emi_scenario': [["E-commerce Shopping", "Home Appliances", "Vehicle", "Personal Loan", "Education"].index(emi_scenario)],
            'requested_amount': [requested_amount],
            'requested_tenure': [requested_tenure],
            'monthly_income': [monthly_income]
        })
        
        # Add derived features
        total_expenses = input_data['travel_expenses'] + input_data['groceries_utilities'] + input_data['other_monthly_expenses']
        input_data['total_expenses'] = total_expenses
        input_data['disposable_income'] = input_data['monthly_income'] - total_expenses - input_data['current_emi_amount']
        input_data['debt_to_income'] = (input_data['current_emi_amount'] + input_data['requested_amount'] / input_data['requested_tenure']) / input_data['monthly_income']
        input_data['expense_to_income'] = total_expenses / input_data['monthly_income']
        input_data['savings_rate'] = (input_data['monthly_income'] - total_expenses - input_data['current_emi_amount']) / input_data['monthly_income']
        input_data['emergency_fund_months'] = input_data['emergency_fund'] / (total_expenses + 1)
        input_data['loan_burden'] = input_data['existing_loans']
        input_data['affordability_score'] = np.clip((input_data['disposable_income'] / (input_data['requested_amount'] / input_data['requested_tenure'])) * 50, 0, 100)
        input_data['risk_score'] = np.clip(100 - (input_data['credit_score'] / 850 * 50) + (input_data['debt_to_income'] * 30), 0, 100)
        
        # Features list
        features = [
            'age', 'gender', 'marital_status', 'education',
            'employment_type', 'years_of_employment', 'company_type',
            'house_type', 'monthly_rent', 'family_size', 'dependents',
            'school_fees', 'college_fees', 'travel_expenses',
            'groceries_utilities', 'other_monthly_expenses',
            'existing_loans', 'current_emi_amount', 'credit_score',
            'bank_balance', 'emergency_fund',
            'emi_scenario', 'requested_amount', 'requested_tenure',
            'monthly_income', 'total_expenses', 'disposable_income',
            'debt_to_income', 'expense_to_income', 'savings_rate',
            'emergency_fund_months', 'loan_burden', 'affordability_score',
            'risk_score'
        ]
        
        X_pred = input_data[features].fillna(0)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if models['classifier'] is not None:
                try:
                    class_pred = models['classifier'].predict(X_pred)[0]
                    class_proba = models['classifier'].predict_proba(X_pred)[0]
                    
                    class_labels = ['Eligible', 'High_Risk', 'Not_Eligible']
                    class_colors = {'Eligible': '✅', 'High_Risk': '⚠️', 'Not_Eligible': '❌'}
                    class_emojis = {'Eligible': '🟢', 'High_Risk': '🟠', 'Not_Eligible': '🔴'}
                    
                    st.metric(
                        "EMI Eligibility",
                        f"{class_emojis.get(class_pred, '')} {class_pred}",
                        delta=f"Confidence: {max(class_proba)*100:.1f}%"
                    )
                    
                    # Show probability distribution
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                        go.Bar(x=class_labels, y=class_proba, 
                              marker_color=['#2ecc71', '#f39c12', '#e74c3c'])
                    ])
                    fig.update_layout(title="Eligibility Probabilities", height=250)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Classification error: {e}")
            else:
                st.warning("⚠️ Classifier model not loaded")
        
        with col2:
            if models['regressor'] is not None:
                try:
                    reg_pred = models['regressor'].predict(X_pred)[0]
                    requested_emi = requested_amount / requested_tenure
                    
                    st.metric(
                        "Maximum Safe Monthly EMI",
                        f"₹{reg_pred:,.0f}",
                        delta=f"Requested EMI: ₹{requested_emi:,.0f}"
                    )
                    
                    # Gauge chart
                    import plotly.graph_objects as go
                    affordability_ratio = reg_pred / requested_emi if requested_emi > 0 else 0
                    gauge_value = min(affordability_ratio, 2) * 50
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = gauge_value,
                        title = {'text': "Affordability Score"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 33], 'color': "#ff6b6b"},
                                {'range': [33, 66], 'color': "#ffd93d"},
                                {'range': [66, 100], 'color': "#6bcb77"}
                            ]
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Regression error: {e}")
            else:
                st.warning("⚠️ Regressor model not loaded")
        
        # Risk Summary
        st.markdown("---")
        st.subheader("📋 Risk Assessment Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Credit Score", credit_score)
        with col2:
            dti = (current_emi + requested_amount/requested_tenure) / monthly_income * 100
            st.metric("Debt-to-Income", f"{dti:.1f}%")
        with col3:
            affordability = (monthly_income - total_expenses - current_emi) / (requested_amount/requested_tenure) * 100
            st.metric("Affordability Ratio", f"{affordability:.1f}%")
        with col4:
            st.metric("Risk Score", f"{input_data['risk_score'].values[0]:.0f}")
        
        # Recommendation
        if 'class_pred' in locals():
            if class_pred == 'Eligible':
                st.success("✅ **Recommendation: Approve Loan** - Customer shows strong financial capacity")
            elif class_pred == 'High_Risk':
                st.warning("⚠️ **Recommendation: Review with Higher Interest** - Marginal case, consider higher rates")
            else:
                st.error("❌ **Recommendation: Decline** - Customer shows high risk of default")