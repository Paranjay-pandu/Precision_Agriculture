import pandas as pd
import numpy as np

# Load the dataset
file_path = 'datasets/Cleaned_TARP.csv'
df = pd.read_csv(file_path)

# Constants
Cn = 900  # constant for the reference crop (grass) in Penman-Monteith equation
Cd = 0.34  # constant for the reference crop (grass) in Penman-Monteith equation
albedo = 0.23  # average albedo for the crop surface
G = 0  # soil heat flux density (assumed to be zero)
gamma = 0.665 * 10**-3 * 101.3  # psychrometric constant (kPa/°C)

# Conversion functions
def es(T):
    # saturation vapor pressure (kPa)
    return 0.6108 * np.exp((17.27 * T) / (T + 237.3))

def ea(RH, es):
    # actual vapor pressure (kPa)
    return RH / 100 * es

def delta(T):
    # slope of the vapor pressure curve (kPa/°C)
    return 4098 * es(T) / (T + 237.3)**2

# Calculate intermediate variables
df['es'] = df['Air temperature (C)'].apply(es)
df['ea'] = df.apply(lambda row: ea(row['Air humidity (%)'], row['es']), axis=1)
df['delta'] = df['Air temperature (C)'].apply(delta)
df['u2'] = df['Wind speed (Km/h)'] * 1000 / 3600  # convert wind speed from km/h to m/s

# Check intermediate calculations
print("Intermediate calculations (first 5 rows):")
print(df[['es', 'ea', 'delta', 'u2']].head())

# Calculate net radiation (R_n) using the simplified formula R_n = (1 - albedo) * solar radiation
# Assuming solar radiation data is not available, we'll use a constant value for demonstration
solar_radiation = 15  # example value (MJ/m^2/day)

# Calculate ET using the Penman-Monteith equation
df['ET'] = (0.408 * df['delta'] * (solar_radiation - G) + gamma * Cn / (df['Air temperature (C)'] + 273) * df['u2'] * (df['es'] - df['ea'])) / (df['delta'] + gamma * (1 + Cd * df['u2']))

# Rename the ET column to Water_predicted
df.rename(columns={'ET': 'Water_predicted'}, inplace=True)

# Check the Water_predicted column
print("Water_predicted column (first 5 rows):")
print(df['Water_predicted'].head())

# Drop intermediate columns used for calculations
df.drop(columns=['es', 'ea', 'delta', 'u2'], inplace=True)

# Save the modified dataset to a new CSV file
modified_file_path = 'Final_Cleaned_TARP.csv'
df.to_csv(modified_file_path, index=False)

print("Modified dataset saved to:", modified_file_path)

