@echo off
title EMIPredict AI Project Runner
color 0A

echo ========================================
echo   🏦 EMIPREDICT AI - Project Runner
echo ========================================
echo.
echo Select an option:
echo.
echo   [1] Generate Dataset (400,000 records)
echo   [2] Train All Models (Full Pipeline)
echo   [3] Run Streamlit Web App
echo   [4] View MLflow Dashboard
echo   [5] Generate Performance Report
echo   [6] Run All (Dataset + Train + Report)
echo   [7] Exit
echo.

set /p choice="Enter your choice (1-7): "

if %choice%==1 (
    echo.
    echo 📊 Generating dataset...
    python src/data_generator.py
    echo.
    echo ✅ Dataset generation complete!
    pause
    goto menu
)

if %choice%==2 (
    echo.
    echo 🚀 Training models...
    python src/train_pipeline.py
    echo.
    echo ✅ Training complete!
    pause
    goto menu
)

if %choice%==3 (
    echo.
    echo 🌐 Starting Streamlit app...
    cd streamlit_app
    streamlit run app.py
    cd ..
    pause
    goto menu
)

if %choice%==4 (
    echo.
    echo 📊 Starting MLflow UI...
    echo Navigate to: http://localhost:5000
    mlflow ui --port 5000 --backend-store-uri ./mlruns
    pause
    goto menu
)

if %choice%==5 (
    echo.
    echo 📄 Generating report...
    python -c "from src.model_training import ModelTrainer; mt = ModelTrainer(); mt.create_model_performance_report(); print('✅ Report generated!')"
    echo.
    echo 📁 Report saved to: reports/model_performance.txt
    pause
    goto menu
)

if %choice%==6 (
    echo.
    echo 🚀 Running full pipeline...
    echo.
    echo Step 1: Generating dataset...
    python src/data_generator.py
    echo.
    echo Step 2: Training models...
    python src/train_pipeline.py
    echo.
    echo Step 3: Generating report...
    python -c "from src.model_training import ModelTrainer; mt = ModelTrainer(); mt.create_model_performance_report(); print('✅ Report generated!')"
    echo.
    echo ✅ All tasks complete!
    echo.
    echo 📁 Check these folders:
    echo    - data/emi_dataset.csv
    echo    - models/*.pkl
    echo    - reports/model_performance.txt
    pause
    goto menu
)

if %choice%==7 (
    echo Goodbye! 👋
    exit
)

:menu
cls
goto top