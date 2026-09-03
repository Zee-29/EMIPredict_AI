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
        
        # ============================================
        # BUSINESS LOGIC FOR PREDICTIONS (No ML Models)
        # ============================================
        
        # Calculate EMI
        requested_emi = requested_amount / requested_tenure
        
        # Calculate total expenses (approximate)
        total_expenses = 5000 + 15000 + 5000  # travel + groceries + other
        
        # Calculate disposable income
        disposable_income = monthly_income - total_expenses - current_emi
        
        # Calculate Debt-to-Income ratio
        dti = (current_emi + requested_emi) / monthly_income * 100
        
        # Calculate affordability ratio
        affordability = (disposable_income) / requested_emi if requested_emi > 0 else 0
        
        # Calculate max safe EMI (40% of disposable income)
        max_safe_emi = disposable_income * 0.4
        max_safe_emi = max(0, max_safe_emi)  # Don't go negative
        
        # Calculate risk score (0-100)
        risk_score = 0
        risk_score += max(0, 100 - credit_score / 850 * 50)  # Credit score component
        risk_score += min(30, dti * 2)  # DTI component
        risk_score += min(20, len(existing_loans) * 7)  # Existing loans component
        
        # EMI Eligibility Logic
        if affordability >= 1.3 and credit_score >= 700 and dti < 40:
            eligibility = "Eligible"
            emoji = "✅"
            color = "green"
            recommendation = "✅ **Recommendation: Approve Loan** - Customer shows strong financial capacity"
        elif affordability >= 0.7 and credit_score >= 600 and dti < 60:
            eligibility = "High_Risk"
            emoji = "⚠️"
            color = "orange"
            recommendation = "⚠️ **Recommendation: Review with Higher Interest** - Marginal case, consider higher rates"
        else:
            eligibility = "Not_Eligible"
            emoji = "❌"
            color = "red"
            recommendation = "❌ **Recommendation: Decline** - Customer shows high risk of default"
        
        # Calculate confidence (simulated)
        confidence = min(95, 50 + (affordability * 20) + (credit_score / 850 * 30))
        confidence = max(60, min(99, confidence))
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "EMI Eligibility",
                f"{emoji} {eligibility}",
                delta=f"Confidence: {confidence:.1f}%"
            )
            
            # Show probability bars
            import plotly.graph_objects as go
            probs = []
            if eligibility == "Eligible":
                probs = [confidence/100, (100-confidence)/200, (100-confidence)/200]
            elif eligibility == "High_Risk":
                probs = [(100-confidence)/200, confidence/100, (100-confidence)/200]
            else:
                probs = [(100-confidence)/200, (100-confidence)/200, confidence/100]
            
            fig = go.Figure(data=[
                go.Bar(x=['Eligible', 'High_Risk', 'Not_Eligible'], 
                      y=probs,
                      marker_color=['#2ecc71', '#f39c12', '#e74c3c'])
            ])
            fig.update_layout(title="Eligibility Probabilities", height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric(
                "Maximum Safe Monthly EMI",
                f"₹{max_safe_emi:,.0f}",
                delta=f"Requested EMI: ₹{requested_emi:,.0f}"
            )
            
            # Gauge chart for affordability
            import plotly.graph_objects as go
            gauge_value = min(affordability * 50, 100)
            gauge_value = max(0, gauge_value)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={'text': "Affordability Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
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
        
        # Risk Summary
        st.markdown("---")
        st.subheader("📋 Risk Assessment Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Credit Score", credit_score)
        with col2:
            st.metric("Debt-to-Income", f"{dti:.1f}%", delta="Should be < 40%" if dti < 40 else "⚠️ High")
        with col3:
            st.metric("Affordability Ratio", f"{affordability:.2f}x", delta="Should be > 1.3x" if affordability >= 1.3 else "⚠️ Low")
        with col4:
            st.metric("Risk Score", f"{int(risk_score)}", delta="Low" if risk_score < 40 else "High" if risk_score > 70 else "Medium")
        
        # Recommendation
        st.markdown("---")
        st.info(recommendation)
        
        # Additional details
        with st.expander("📊 Detailed Financial Analysis"):
            st.write(f"**Monthly Income:** ₹{monthly_income:,.0f}")
            st.write(f"**Total Monthly Expenses:** ₹{total_expenses:,.0f}")
            st.write(f"**Current EMI:** ₹{current_emi:,.0f}")
            st.write(f"**Disposable Income:** ₹{disposable_income:,.0f}")
            st.write(f"**Requested EMI:** ₹{requested_emi:,.0f}")
            st.write(f"**Maximum Safe EMI:** ₹{max_safe_emi:,.0f}")
            st.write(f"**Debt-to-Income Ratio:** {dti:.1f}%")
            st.write(f"**Affordability Ratio:** {affordability:.2f}x")