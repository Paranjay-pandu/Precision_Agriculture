import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv('datasets/final.csv')
columns_to_remove = ['Water_predicted', 'Time', 'Wind gust (Km/h)', 'Pressure (KPa)', 'N', 'P', 'K']
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns_to_remove, axis=1)
y = df['Water_predicted']

# Split data and standardize features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR()
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    cv_score = cross_val_score(model, X, y, cv=5).mean()
    
    results.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'R^2': r2,
        'CV Score': cv_score
    })

results_df = pd.DataFrame(results)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Performance Comparison')

metrics = ['MSE', 'RMSE', 'R^2', 'CV Score']
colors = ['blue', 'green', 'red', 'purple']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[i // 2, i % 2]
    ax.bar(results_df['Model'], results_df[metric], color=color)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()