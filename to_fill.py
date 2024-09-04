import pandas as pd

# Load the dataset
file_path = 'datasets/TARP.csv'
df = pd.read_csv(file_path)

# Remove leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Fill missing values
# For columns with a lot of missing values, we can fill them with the mean (or median)
# If the columns are not critical, we could also drop them
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

# Display the cleaned dataset
df.info()
df.head()

# Save the cleaned dataset
cleaned_file_path = '/mnt/data/Cleaned_TARP.csv'
df.to_csv(cleaned_file_path, index=False)
