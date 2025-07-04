# Housing Price Prediction using Regression
# Housing Price Regression

This project performs housing price prediction using three regression models and compares their performance based on MSE and R² score.

## 📁 Folder Structure
HousingRegression/
│
├── regression.py # Main Python script to run all models
├── requirements.txt # Python dependencies
├── .github/
│ └── workflows/
│ └── regression_workflow.yml # GitHub Actions workflow
└── README.md # This file


## ⚙️ Models Compared

1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**

## 📊 Evaluation Metrics

| Model              | MSE     | R² Score |
|--------------------|---------|----------|
| Linear Regression  | 24.29   | 0.67     |
| Ridge Regression   | 24.48   | 0.67     |
| Lasso Regression   | 25.16   | 0.66     |

## 🚀 How to Run

```bash
# Create conda environment
conda create -n housing python=3.10 -y
conda activate housing

# Clone and enter project
git clone https://github.com/g24ai1102/HousingRegression.git
cd HousingRegression

# Install requirements
pip install -r requirements.txt

# Run regression script
python regression.py

GitHub Actions
A GitHub Actions CI pipeline is triggered on every push:

Installs dependencies

Runs regression.py

Logs outputs (MSE & R²) in workflow run

You can check the CI logs at:
GitHub Actions → https://github.com/g24ai1102/HousingRegression/actions


Used Boston Housing Dataset from CMU’s original source due to deprecation of load_boston in scikit-learn.
data_url = "http://lib.stat.cmu.edu/datasets/boston"


Dependencies
See requirements.txt

Author
Umang Garg

GitHub: https://github.com/g24ai1102
