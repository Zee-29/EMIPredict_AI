import numpy as np
import pandas as pd
from datetime import datetime
import os

class EMIDataGenerator:
    """Generate synthetic financial dataset with 400,000 records"""
    
    def __init__(self, num_records=400000, seed=42):
        self.num_records = num_records
        np.random.seed(seed)
        
        # EMI Scenarios and their parameters
        self.emi_scenarios = {
            'E-commerce Shopping': {'min_amount': 10000, 'max_amount': 200000, 'min_tenure': 3, 'max_tenure': 24},
            'Home Appliances': {'min_amount': 20000, 'max_amount': 300000, 'min_tenure': 6, 'max_tenure': 36},
            'Vehicle': {'min_amount': 80000, 'max_amount': 1500000, 'min_tenure': 12, 'max_tenure': 84},
            'Personal Loan': {'min_amount': 50000, 'max_amount': 1000000, 'min_tenure': 12, 'max_tenure': 60},
            'Education': {'min_amount': 50000, 'max_amount': 500000, 'min_tenure': 6, 'max_tenure': 48}
        }
        
        self.education_levels = ['High School', 'Graduate', 'Post Graduate', 'Professional']
        self.employment_types = ['Private', 'Government', 'Self-employed']
        self.company_types = ['Startup', 'SME', 'Large Corporate', 'MNC']
        self.genders = ['Male', 'Female']
        self.marital_status = ['Single', 'Married', 'Divorced']
        self.house_types = ['Rented', 'Own', 'Family']
        
    def generate_personal_demographics(self, n):
        """Generate personal demographic features"""
        return {
            'age': np.random.randint(25, 60, n),
            'gender': np.random.choice(self.genders, n),
            'marital_status': np.random.choice(self.marital_status, n, p=[0.35, 0.55, 0.10]),
            'education': np.random.choice(self.education_levels, n, p=[0.20, 0.40, 0.30, 0.10])
        }
    
    def generate_employment(self, n):
        """Generate employment features"""
        employment = np.random.choice(self.employment_types, n, p=[0.50, 0.25, 0.25])
        years_exp = np.random.randint(0, 35, n)
        
        # Correlate employment type with experience
        for i in range(n):
            if employment[i] == 'Self-employed':
                years_exp[i] = np.random.randint(2, 30)
            elif employment[i] == 'Government':
                years_exp[i] = np.random.randint(1, 35)
        
        return {
            'employment_type': employment,
            'years_of_employment': years_exp,
            'company_type': np.random.choice(self.company_types, n, p=[0.15, 0.25, 0.35, 0.25])
        }
    
    def generate_housing_family(self, n):
        """Generate housing and family features"""
        family_size = np.random.randint(1, 7, n)
        dependents = np.random.randint(0, 4, n)
        # Dependents should be <= family_size - 2 (assuming 2 adults)
        dependents = np.minimum(dependents, family_size - 1)
        dependents = np.maximum(dependents, 0)
        
        return {
            'house_type': np.random.choice(self.house_types, n, p=[0.30, 0.45, 0.25]),
            'monthly_rent': np.random.randint(0, 50000, n),
            'family_size': family_size,
            'dependents': dependents
        }
    
    def generate_expenses(self, n):
        """Generate monthly expense features"""
        return {
            'school_fees': np.random.randint(0, 30000, n),
            'college_fees': np.random.randint(0, 50000, n),
            'travel_expenses': np.random.randint(1000, 20000, n),
            'groceries_utilities': np.random.randint(3000, 40000, n),
            'other_monthly_expenses': np.random.randint(1000, 30000, n)
        }
    
    def generate_financial_status(self, n):
        """Generate financial status and credit features"""
        credit_scores = np.random.normal(680, 100, n)
        credit_scores = np.clip(credit_scores, 300, 850).astype(int)
        
        return {
            'existing_loans': np.random.choice(['None', '1', '2', '3+'], n, p=[0.40, 0.30, 0.20, 0.10]),
            'current_emi_amount': np.random.randint(0, 30000, n),
            'credit_score': credit_scores,
            'bank_balance': np.random.randint(10000, 500000, n),
            'emergency_fund': np.random.randint(5000, 300000, n)
        }
    
    def generate_loan_details(self, n):
        """Generate loan application details with scenario-based distributions"""
        scenarios = list(self.emi_scenarios.keys())
        chosen_scenarios = np.random.choice(scenarios, n, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        requested_amounts = []
        requested_tenures = []
        
        for scenario in chosen_scenarios:
            params = self.emi_scenarios[scenario]
            amount = np.random.randint(params['min_amount'], params['max_amount'])
            tenure = np.random.randint(params['min_tenure'], params['max_tenure'])
            requested_amounts.append(amount)
            requested_tenures.append(tenure)
        
        return {
            'emi_scenario': chosen_scenarios,
            'requested_amount': np.array(requested_amounts),
            'requested_tenure': np.array(requested_tenures)
        }
    
    def calculate_monthly_income(self, age, employment_type, years_exp, education):
        """Calculate monthly income based on profile"""
        base_income = 20000
        
        # Age factor
        if age < 30:
            base_income += np.random.randint(5000, 15000)
        elif age < 40:
            base_income += np.random.randint(15000, 35000)
        else:
            base_income += np.random.randint(20000, 40000)
        
        # Education factor
        edu_multipliers = {'High School': 1.0, 'Graduate': 1.3, 'Post Graduate': 1.6, 'Professional': 2.0}
        base_income *= edu_multipliers.get(education, 1.0)
        
        # Employment factor
        emp_multipliers = {'Private': 1.0, 'Government': 1.2, 'Self-employed': 1.4}
        base_income *= emp_multipliers.get(employment_type, 1.0)
        
        # Experience factor
        base_income *= (1 + years_exp * 0.02)
        
        return int(base_income + np.random.randint(-5000, 5000))
    
    def calculate_targets(self, df):
        """Calculate eligibility and max EMI targets"""
        # Calculate monthly income
        monthly_income = []
        for idx, row in df.iterrows():
            income = self.calculate_monthly_income(
                row['age'], row['employment_type'], 
                row['years_of_employment'], row['education']
            )
            monthly_income.append(income)
        
        df['monthly_income'] = monthly_income
        
        # Calculate total monthly expenses
        df['total_expenses'] = (
            df['school_fees'] + df['college_fees'] + 
            df['travel_expenses'] + df['groceries_utilities'] + 
            df['other_monthly_expenses'] + df['monthly_rent']
        )
        
        # Calculate disposable income
        df['disposable_income'] = df['monthly_income'] - df['total_expenses'] - df['current_emi_amount']
        
        # Calculate debt-to-income ratio
        df['debt_to_income'] = (df['current_emi_amount'] + df['requested_amount'] / df['requested_tenure']) / df['monthly_income']
        
        # Calculate affordability ratio
        df['affordability_ratio'] = df['disposable_income'] / df['monthly_income']
        
        # Calculate max monthly EMI (regression target)
        # Rule: Max EMI = 40% of disposable income, capped by income considerations
        df['max_monthly_emi'] = np.maximum(
            500,
            np.minimum(
                df['disposable_income'] * 0.4,
                50000
            )
        ).astype(int)
        
        # Adjust based on credit score and existing loans
        credit_multiplier = np.where(df['credit_score'] > 700, 1.2, 
                                    np.where(df['credit_score'] > 600, 1.0, 0.7))
        
        existing_loan_penalty = np.where(df['existing_loans'] == 'None', 1.0,
                                        np.where(df['existing_loans'] == '1', 0.9,
                                                np.where(df['existing_loans'] == '2', 0.7, 0.5)))
        
        df['max_monthly_emi'] = (df['max_monthly_emi'] * credit_multiplier * existing_loan_penalty).astype(int)
        df['max_monthly_emi'] = np.maximum(500, df['max_monthly_emi'])
        df['max_monthly_emi'] = np.minimum(50000, df['max_monthly_emi'])
        
        # Calculate EMI eligibility (classification target - 3 classes)
        requested_emi = df['requested_amount'] / df['requested_tenure']
        affordability = df['max_monthly_emi'] / requested_emi
        
        df['emi_eligibility'] = np.where(
            affordability >= 1.3, 'Eligible',
            np.where(affordability >= 0.7, 'High_Risk', 'Not_Eligible')
        )
        
        return df
    
    def generate_dataset(self, output_path='data/emi_dataset.csv'):
        """Generate complete dataset with all features and targets"""
        print("Generating 400,000 financial records...")
        n = self.num_records
        
        # Generate all features
        data = {}
        data.update(self.generate_personal_demographics(n))
        data.update(self.generate_employment(n))
        data.update(self.generate_housing_family(n))
        data.update(self.generate_expenses(n))
        data.update(self.generate_financial_status(n))
        data.update(self.generate_loan_details(n))
        
        df = pd.DataFrame(data)
        
        # Calculate targets
        df = self.calculate_targets(df)
        
        # Select final columns (22 features + 2 targets)
        feature_columns = [
            'age', 'gender', 'marital_status', 'education',
            'employment_type', 'years_of_employment', 'company_type',
            'house_type', 'monthly_rent', 'family_size', 'dependents',
            'school_fees', 'college_fees', 'travel_expenses', 
            'groceries_utilities', 'other_monthly_expenses',
            'existing_loans', 'current_emi_amount', 'credit_score',
            'bank_balance', 'emergency_fund',
            'emi_scenario', 'requested_amount', 'requested_tenure',
            # Derived features
            'monthly_income', 'total_expenses', 'disposable_income',
            'debt_to_income', 'affordability_ratio'
        ]
        
        # Keep only required columns
        df = df[[col for col in feature_columns if col in df.columns] + 
                ['emi_eligibility', 'max_monthly_emi']]
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save dataset
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset generated with {len(df)} records")
        print(f"📁 Saved to: {output_path}")
        
        # Display summary
        print("\n📊 Dataset Summary:")
        print(f"  - Total Records: {len(df):,}")
        print(f"  - Features: {len(df.columns) - 2}")
        print(f"  - EMI Eligibility Distribution:")
        print(df['emi_eligibility'].value_counts())
        print(f"\n  - Max Monthly EMI Statistics:")
        print(df['max_monthly_emi'].describe())
        
        return df

if __name__ == "__main__":
    generator = EMIDataGenerator(num_records=400000)
    df = generator.generate_dataset()