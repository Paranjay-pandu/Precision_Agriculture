from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv('datasets/final.csv')

# Specify columns to remove
columns_to_remove = ['Water_predicted', 'Time', 'Wind gust (Km/h)', 'Pressure (KPa)', 'N', 'P', 'K']

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split data into features and target
x = data.drop(columns=columns_to_remove)
y = data['Water_predicted']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train, y_train)

# Predict and evaluate the model
y_pred = rf.predict(x_test)
r_score = r2_score(y_test, y_pred)

print(r_score)