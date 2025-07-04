import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load data manually as per assignment instruction
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # Merge alternating rows for full feature vectors
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

# Evaluate model performance
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse:.2f}, R²: {r2:.2f}")
    return name, mse, r2

# Load and split data
df = load_data()
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# Linear Regression (no major hyperparams, use as baseline)
lr = LinearRegression()
results.append(evaluate_model("Linear Regression", lr, X_train, X_test, y_train, y_test))

# Ridge Regression with GridSearchCV
ridge_params = {
    'alpha': [0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky']
}
ridge = Ridge()
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5)
results.append(evaluate_model("Ridge Regression (Tuned)", ridge_grid, X_train, X_test, y_train, y_test))

# Lasso Regression with GridSearchCV
lasso_params = {
    'alpha': [0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
    'max_iter': [1000, 5000, 10000]
}
lasso = Lasso()
lasso_grid = GridSearchCV(lasso, lasso_params, cv=5)
results.append(evaluate_model("Lasso Regression (Tuned)", lasso_grid, X_train, X_test, y_train, y_test))