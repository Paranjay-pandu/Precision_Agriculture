import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

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

# Save the trained model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(rf, file)

print("Model saved as model.pkl")
