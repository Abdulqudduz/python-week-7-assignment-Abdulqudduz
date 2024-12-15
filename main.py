#multiplot in one figure

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData types and missing values:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean dataset (if needed)
if df.isnull().sum().sum() > 0:
    df.dropna(inplace=True)
    print("\nMissing values dropped.")
else:
    print("\nNo missing values detected.")

# Compute basic statistics
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Compute mean of numerical columns grouped by species
grouped_means = df.groupby('target').mean()
print("\nMean of numerical columns grouped by species:")
print(grouped_means)

# Combine all plots into one figure
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Create a 2x2 grid of subplots

# Line Chart: Trend of Petal Length Over Samples
axes[0, 0].plot(df.index, df['petal length (cm)'], label='Petal Length', color='blue')
axes[0, 0].set_title("Trend of Petal Length Over Samples")
axes[0, 0].set_xlabel("Sample Index")
axes[0, 0].set_ylabel("Petal Length (cm)")
axes[0, 0].legend()

# Bar Chart: Average Petal Length per Species
species_means = df.groupby('target')['petal length (cm)'].mean()
axes[0, 1].bar(species_means.index, species_means, color=['red', 'green', 'blue'])
axes[0, 1].set_title("Average Petal Length per Species")
axes[0, 1].set_xlabel("Species")
axes[0, 1].set_ylabel("Average Petal Length (cm)")


# Histogram: Distribution of Sepal Length
axes[1, 0].hist(df['sepal length (cm)'], bins=20, color='purple', edgecolor='black')
axes[1, 0].set_title("Distribution of Sepal Length")
axes[1, 0].set_xlabel("Sepal Length (cm)")
axes[1, 0].set_ylabel("Frequency")

# Scatter Plot: Sepal Length vs Petal Length
scatter = axes[1, 1].scatter(
    df['sepal length (cm)'],
    df['petal length (cm)'],
    c=df['target'],
    cmap='viridis'
)
axes[1, 1].set_title("Sepal Length vs Petal Length")
axes[1, 1].set_xlabel("Sepal Length (cm)")
axes[1, 1].set_ylabel("Petal Length (cm)")
fig.colorbar(scatter, ax=axes[1, 1], label="Species")

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the combined plots
plt.show()