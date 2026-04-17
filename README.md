# Diabetes Prediction Project

This repository contains a Streamlit app and notebook for diabetes prediction.

## Project Files

- app.py: Streamlit app
- requirements.txt: Python dependencies
- dataset/diabetes.csv: dataset used by the project
- notebook/diabeties_pred.ipynb: model training notebook
- notebook/models/: saved model files used by app.py

## Run Locally

1. Clone the repository.
2. Open terminal in the project root.
3. Create a virtual environment:
   - Windows: python -m venv .venv
   - macOS/Linux: python3 -m venv .venv
4. Activate virtual environment:
   - Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
   - Windows CMD: .\\.venv\\Scripts\\activate.bat
   - macOS/Linux: source .venv/bin/activate
5. Install dependencies:
   - pip install -r requirements.txt
6. Run the app:
   - streamlit run app.py

## Push To GitHub

If this is your first push:

1. git init
2. git add .
3. git commit -m "Initial commit"
4. git branch -M main
5. git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
6. git push -u origin main

## Notes

- Keep model files in notebook/models/ so app.py can load them.
- This project is for educational use.
