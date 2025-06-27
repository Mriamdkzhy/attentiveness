import pandas as pd

# Load the CSV file for a particular model
df = pd.read_csv('attentive.csv')  # Replace with your actual filename

df['Attentiveness Score'] = pd.to_numeric(df['Attentiveness Score'], errors='coerce')

# Group by model name and calculate average
stats = df.groupby('Model')['Attentiveness Score'].agg(['mean', 'std']).reset_index()

# Print results
print("Average Attentiveness Scores by Model:")
print(stats)