import pandas as pd
from sklearn.datasets import load_diabetes

# Load the Diabetes dataset
diabetes = load_diabetes()

# Create a DataFrame from the dataset
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Save DataFrame to CSV
df.to_csv('data/diabetes.csv', index=False)
