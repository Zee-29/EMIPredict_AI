import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    """Feature engineering pipeline for EMI dataset"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.categorical_cols = []
        self.numerical_cols = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def identify_column_types(self):
        """Identify categorical and numerical columns"""
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns from features
        if 'emi_eligibility' in self.categorical_cols:
            self.categorical_cols.remove('emi_eligibility')
        if 'max_monthly_emi' in self.numerical_cols:
            self.numerical_cols.remove('max_monthly_emi')
            
        print(f"Categorical features: {len(self.categorical_cols)}")
        print(f"Numerical features: {len(self.numerical_cols)}")
        
    def create_derived_features(self):
        """Create derived financial ratios and interaction features"""
        
        # Debt-to-Income Ratio (already exists but we'll create additional)
        if 'debt_to_income' not in self.df.columns:
            self.df['debt_to_income'] = self.df['current_emi_amount'] / self.df['monthly_income']
        
        # Expense-to-Income Ratio
        self.df['expense_to_income'] = self.df['total_expenses'] / self.df['monthly_income']
        
        # Savings Rate
        self.df['savings_rate'] = (self.df['monthly_income'] - self.df['total_expenses'] - 
                                   self.df['current_emi_amount']) / self.df['monthly_income']
        
        # Credit Score Category
        self.df['credit_score_category'] = pd.cut(
            self.df['credit_score'],
            bins=[0, 580, 669, 739, 799, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        # Emergency Fund Coverage (months)
        self.df['emergency_fund_months'] = self.df['emergency_fund'] / (self.df['total_expenses'] + 1)
        
        # Existing Loan Burden
        self.df['loan_burden'] = np.where(
            self.df['existing_loans'] == 'None', 0,
            np.where(self.df['existing_loans'] == '1', 1,
                    np.where(self.df['existing_loans'] == '2', 2, 3))
        )
        
        # Age Category
        self.df['age_category'] = pd.cut(
            self.df['age'],
            bins=[0, 30, 40, 50, 100],
            labels=['Young', 'Early Career', 'Mid Career', 'Senior']
        )
        
        # Affordability Score (0-100)
        self.df['affordability_score'] = np.clip(
            (self.df['max_monthly_emi'] / (self.df['requested_amount'] / self.df['requested_tenure'])) * 50,
            0, 100
        ).astype(int)
        
        # Risk Score (0-100)
        self.df['risk_score'] = np.clip(
            100 - (self.df['credit_score'] / 850 * 50) + (self.df['debt_to_income'] * 30),
            0, 100
        ).astype(int)
        
        print("✅ Created derived features")
        return self.df
    
    def encode_categorical(self):
        """Encode categorical variables using Label Encoding"""
        for col in self.categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   Encoded: {col}")
        
        # Also encode newly created categorical features
        for col in ['credit_score_category', 'age_category']:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   Encoded: {col}")
                
        return self.df
    
    def prepare_features(self):
        """Prepare final feature set for modeling"""
        # Define all features to use
        feature_cols = [
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
        
        # Ensure all features exist
        available_features = [col for col in feature_cols if col in self.df.columns]
        missing = set(feature_cols) - set(available_features)
        if missing:
            print(f"⚠️ Missing features: {missing}")
        
        # Prepare feature matrix
        X = self.df[available_features].copy()
        
        # Encode categorical
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numerical features
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X[self.numerical_cols] = self.scaler.fit_transform(X[self.numerical_cols])
        
        # Targets
        y_class = self.df['emi_eligibility']
        y_reg = self.df['max_monthly_emi']
        
        print(f"\n✅ Feature matrix shape: {X.shape}")
        print(f"   Features: {X.columns.tolist()}")
        
        return X, y_class, y_reg
    
    def split_data(self, X, y_class, y_reg, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_train_val, X_test, y_class_train_val, y_class_test = train_test_split(
            X, y_class, test_size=test_size, random_state=random_state, stratify=y_class
        )
        _, _, y_reg_train_val, y_reg_test = train_test_split(
            X, y_reg, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_class_train, y_class_val = train_test_split(
            X_train_val, y_class_train_val, 
            test_size=val_size_adjusted, random_state=random_state, stratify=y_class_train_val
        )
        _, _, y_reg_train, y_reg_val = train_test_split(
            X_train_val, y_reg_train_val,
            test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_class_train': y_class_train, 'y_class_val': y_class_val, 'y_class_test': y_class_test,
            'y_reg_train': y_reg_train, 'y_reg_val': y_reg_val, 'y_reg_test': y_reg_test
        }