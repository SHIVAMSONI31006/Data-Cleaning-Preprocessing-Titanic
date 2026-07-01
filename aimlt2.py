import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

np.random.seed(42)
data_size = 150
dummy_data = {
    'Age': np.random.randint(18, 70, size=data_size),
    'Income': np.random.exponential(scale=50000, size=data_size) + 20000,
    'Savings': np.random.normal(loc=15000, scale=500, size=data_size),
    'Department': np.random.choice(['Tech', 'Sales', 'HR', 'Marketing'], size=data_size),
    'Performance_Score': np.random.uniform(1, 5, size=data_size)
}
df = pd.DataFrame(dummy_data)

df.loc[10, 'Age'] = 135  
df.loc[25, 'Income'] = np.nan  
df.loc[50, 'Income'] = np.nan  

print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution & Outlier Analysis (Numeric Features)', fontsize=16, fontweight='bold')

sns.histplot(df['Income'], kde=True, bins=25, color='royalblue', ax=axes[0, 0])
axes[0, 0].set_title('Income Distribution')

sns.histplot(df['Savings'], kde=True, bins=25, color='forestgreen', ax=axes[0, 1])
axes[0, 1].set_title('Savings Distribution')

sns.boxplot(x=df['Age'], color='darkorange', ax=axes[1, 0])
axes[1, 0].set_title('Age Boxplot')

sns.boxplot(x=df['Performance_Score'], color='crimson', ax=axes[1, 1])
axes[1, 1].set_title('Performance Score Boxplot')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number]) 
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
plt.title('Feature Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.show()

sns.pairplot(df, hue='Department', palette='Set2', diag_kind='kde')
plt.show()

fig_interactive = px.scatter(
    df, 
    x="Income", 
    y="Savings", 
    color="Department", 
    size="Performance_Score",
    hover_data=['Age'],
    title="Interactive Analysis: Income vs Savings by Department"
)
fig_interactive.show()