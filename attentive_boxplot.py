import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('attentive.csv')  # Replace with your actual filename

# Ensure numeric values
df['Attentiveness Score'] = pd.to_numeric(df['Attentiveness Score'], errors='coerce')

# Drop rows with missing values
df.dropna(subset=['Attentiveness Score', 'Model'], inplace=True)

# Create the boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='Model',
    y='Attentiveness Score',
    data=df,
    medianprops={"color": "red", "linewidth": 2}
)


# Customize the plot
plt.title('Boxplot of Attentiveness Score by Model', fontsize=25)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Attentive Score', fontsize=25)
plt.xlabel('Model', fontsize=25)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('attentiveness_score_boxplot.png', dpi=300)
plt.show()
