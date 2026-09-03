import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and track multiple models using MLflow"""
    
    def __init__(self, experiment_name="EMIPredict_AI"):
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Set MLflow
        mlflow.set_experiment(experiment_name)
        
    def train_classification_models(self, X_train, y_train, X_val, y_val):
        """Train classification models for EMI eligibility"""
        print("\n" + "="*60)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*60)
        
        # Encode string labels to numeric for XGBoost
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Get class names for reference
        class_names = self.label_encoder.classes_
        print(f"   Classes: {class_names}")
        
        # Model configurations with class balancing
        models = {
            'Logistic_Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=50, 
                random_state=42, 
                n_jobs=-1,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=50, 
                learning_rate=0.1, 
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )
        }
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            with mlflow.start_run(run_name=f"Classification_{name}"):
                # Log parameters
                params = model.get_params() if hasattr(model, 'get_params') else {}
                mlflow.log_params(params)
                
                # For XGBoost, use encoded labels; for others, use original
                if name == 'XGBoost':
                    model.fit(X_train, y_train_encoded)
                    y_pred_encoded = model.predict(X_val)
                    y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
                    y_proba = model.predict_proba(X_val)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics (use original string labels for consistent evaluation)
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
                    'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0)
                }
                
                if y_proba is not None:
                    try:
                        if len(class_names) == 2:
                            metrics['roc_auc'] = roc_auc_score(y_val_encoded, y_proba[:, 1])
                        else:
                            metrics['roc_auc'] = roc_auc_score(y_val_encoded, y_proba, multi_class='ovr')
                    except:
                        metrics['roc_auc'] = 0.0
                
                mlflow.log_metrics(metrics)
                
                # Log model
                if name == 'XGBoost':
                    mlflow.xgboost.log_model(model, f"model_{name}")
                else:
                    mlflow.sklearn.log_model(model, f"model_{name}")
                
                # Save locally
                joblib.dump(model, f'models/{name}_classifier.pkl')
                
                # Store results
                self.models[f'class_{name}'] = model
                self.results[f'class_{name}'] = metrics
                
                print(f"   ✅ {name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
        
        # Save label encoder for later use
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        return self.results

    def train_regression_models(self, X_train, y_train, X_val, y_val):
        """Train regression models for max monthly EMI prediction"""
        print("\n" + "="*60)
        print("TRAINING REGRESSION MODELS")
        print("="*60)
        
        models = {
            'Linear_Regression': LinearRegression(),
            'Random_Forest': RandomForestRegressor(
                n_estimators=50, 
                random_state=42, 
                n_jobs=-1,
                min_samples_split=10
            ),
            'XGBoost': XGBRegressor(
                n_estimators=50, 
                learning_rate=0.1, 
                random_state=42,
                verbosity=0
            )
        }
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            with mlflow.start_run(run_name=f"Regression_{name}"):
                params = model.get_params() if hasattr(model, 'get_params') else {}
                mlflow.log_params(params)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_val, y_pred)
                mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1))) * 100
                
                metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
                
                mlflow.log_metrics(metrics)
                
                # Log model
                if name == 'XGBoost':
                    mlflow.xgboost.log_model(model, f"model_{name}")
                else:
                    mlflow.sklearn.log_model(model, f"model_{name}")
                
                joblib.dump(model, f'models/{name}_regressor.pkl')
                
                self.models[f'reg_{name}'] = model
                self.results[f'reg_{name}'] = metrics
                
                print(f"   ✅ {name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return self.results
    
    def select_best_models(self):
        """Select best performing models"""
        print("\n" + "="*60)
        print("🏆 BEST MODEL SELECTION")
        print("="*60)
        
        best_class = None
        best_reg = None
        
        # Classification: best by accuracy
        class_results = {k: v for k, v in self.results.items() if k.startswith('class_')}
        if class_results:
            best_class = max(class_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\n🏆 Best Classification Model: {best_class[0]}")
            print(f"   Accuracy: {best_class[1]['accuracy']:.4f}")
            print(f"   F1 Score: {best_class[1]['f1_macro']:.4f}")
        
        # Regression: best by R²
        reg_results = {k: v for k, v in self.results.items() if k.startswith('reg_')}
        if reg_results:
            best_reg = max(reg_results.items(), key=lambda x: x[1]['r2'])
            print(f"\n🏆 Best Regression Model: {best_reg[0]}")
            print(f"   R² Score: {best_reg[1]['r2']:.4f}")
            print(f"   RMSE: {best_reg[1]['rmse']:.2f}")
        
        return best_class, best_reg

    def create_model_performance_report(self):
        """Generate and save model performance report"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        text_content = self._generate_text_report()
        
        with open(f'reports/model_performance_{timestamp}.txt', 'w') as f:
            f.write(text_content)
        with open('reports/model_performance.txt', 'w') as f:
            f.write(text_content)
        
        print(f"✅ Reports saved to reports/")
        return timestamp
    
    def _generate_text_report(self):
        """Generate text report"""
        lines = []
        lines.append("="*70)
        lines.append("  EMIPREDICT AI - MODEL PERFORMANCE REPORT")
        lines.append("="*70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*70)
        
        lines.append("\nCLASSIFICATION MODELS")
        lines.append("-"*40)
        for name, metrics in sorted(self.results.items()):
            if name.startswith('class_'):
                lines.append(f"\n{name.replace('class_', '')}:")
                for metric, value in metrics.items():
                    lines.append(f"  {metric}: {value:.4f}")
        
        lines.append("\nREGRESSION MODELS")
        lines.append("-"*40)
        for name, metrics in sorted(self.results.items()):
            if name.startswith('reg_'):
                lines.append(f"\n{name.replace('reg_', '')}:")
                lines.append(f"  rmse: {metrics.get('rmse', 0):.2f}")
                lines.append(f"  mae: {metrics.get('mae', 0):.2f}")
                lines.append(f"  r2: {metrics.get('r2', 0):.4f}")
                lines.append(f"  mape: {metrics.get('mape', 0):.2f}%")
        
        lines.append("\n" + "="*70)
        lines.append("END OF REPORT")
        lines.append("="*70)
        
        return '\n'.join(lines)