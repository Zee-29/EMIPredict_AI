import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import EMIDataGenerator
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.mlflow_setup import setup_mlflow

def run_training_pipeline():
    """Complete training pipeline"""
    print("="*70)
    print("🚀 EMIPREDICT AI - TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Generate/Load Data
    print("\n📊 Step 1: Data Generation/Loading")
    if os.path.exists('data/emi_dataset.csv'):
        df = pd.read_csv('data/emi_dataset.csv')
        print(f"✅ Loaded existing dataset: {len(df)} records")
    else:
        print("Generating new dataset...")
        generator = EMIDataGenerator(num_records=400000)
        df = generator.generate_dataset()
    
    # Step 2: Feature Engineering
    print("\n🔧 Step 2: Feature Engineering")
    engineer = FeatureEngineer(df)
    engineer.identify_column_types()
    engineer.create_derived_features()
    X, y_class, y_reg = engineer.prepare_features()
    
    # Step 3: Data Split
    print("\n📊 Step 3: Data Split")
    data = engineer.split_data(X, y_class, y_reg)
    
    # Step 4: Setup MLflow
    print("\n📊 Step 4: MLflow Setup")
    experiment_name = setup_mlflow()
    
    # Step 5: Train Classification Models
    print("\n📊 Step 5: Training Classification Models")
    trainer = ModelTrainer(experiment_name)
    class_results = trainer.train_classification_models(
        data['X_train'], data['y_class_train'],
        data['X_val'], data['y_class_val']
    )
    
    # Step 6: Train Regression Models
    print("\n📊 Step 6: Training Regression Models")
    reg_results = trainer.train_regression_models(
        data['X_train'], data['y_reg_train'],
        data['X_val'], data['y_reg_val']
    )
    
    # Step 7: Select Best Models
    print("\n📊 Step 7: Selecting Best Models")
    best_class, best_reg = trainer.select_best_models()
    
    # Step 8: Test on Test Set
    print("\n📊 Step 8: Final Test Evaluation")
    if best_class:
        model = trainer.models[best_class[0]]
        y_pred = model.predict(data['X_test'])
        from sklearn.metrics import accuracy_score, f1_score
        test_acc = accuracy_score(data['y_class_test'], y_pred)
        test_f1 = f1_score(data['y_class_test'], y_pred, average='macro')
        print(f"   ✅ {best_class[0]} Test Accuracy: {test_acc:.4f}")
        print(f"   ✅ {best_class[0]} Test F1: {test_f1:.4f}")
    
    if best_reg:
        model = trainer.models[best_reg[0]]
        y_pred = model.predict(data['X_test'])
        from sklearn.metrics import mean_squared_error, r2_score
        test_rmse = np.sqrt(mean_squared_error(data['y_reg_test'], y_pred))
        test_r2 = r2_score(data['y_reg_test'], y_pred)
        print(f"   ✅ {best_reg[0]} Test RMSE: {test_rmse:.2f}")
        print(f"   ✅ {best_reg[0]} Test R²: {test_r2:.4f}")
    
    # Step 9: Save Best Models
    print("\n📊 Step 9: Saving Best Models")
    import joblib
    
    if best_class:
        joblib.dump(trainer.models[best_class[0]], 'models/best_classifier.pkl')
        print(f"   ✅ Best classifier saved: models/best_classifier.pkl")
    
    if best_reg:
        joblib.dump(trainer.models[best_reg[0]], 'models/best_regressor.pkl')
        print(f"   ✅ Best regressor saved: models/best_regressor.pkl")
    
    # Step 10: Generate Report
    print("\n📊 Step 10: Generating Reports")
    trainer.create_model_performance_report()
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    
    print("\n📁 Output Files:")
    print("   - models/best_classifier.pkl (Best classification model)")
    print("   - models/best_regressor.pkl (Best regression model)")
    print("   - reports/model_performance.txt (Performance report)")
    print("   - mlflow_experiments/ (MLflow tracking data)")
    
    print("\n🌐 To view MLflow UI: mlflow ui --port 5000")

if __name__ == "__main__":
    run_training_pipeline()