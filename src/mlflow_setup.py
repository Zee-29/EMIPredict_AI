import mlflow
import os
import subprocess
from pathlib import Path

def setup_mlflow():
    """Setup MLflow with file-based storage"""
    print("🔧 Setting up MLflow...")
    
    # ✅ Allow file store backend (fixes the error)
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    
    # Create directories
    Path('mlflow_experiments').mkdir(exist_ok=True)
    Path('mlruns').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('reports').mkdir(exist_ok=True)
    
    # Set tracking URI
    mlflow.set_tracking_uri('./mlruns')
    
    # Create experiment
    experiment_name = "EMIPredict_AI"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✅ Using existing experiment: {experiment_name}")
    except Exception as e:
        print(f"⚠️ Using default experiment: {e}")
    
    return experiment_name

def start_mlflow_ui():
    """Start MLflow UI"""
    print("\n🌐 Starting MLflow UI...")
    print("   Navigate to: http://localhost:5000")
    
    try:
        subprocess.run([
            'mlflow', 'ui',
            '--port', '5000',
            '--backend-store-uri', './mlruns'
        ])
    except KeyboardInterrupt:
        print("\n⏹️ MLflow UI stopped")

if __name__ == "__main__":
    setup_mlflow()