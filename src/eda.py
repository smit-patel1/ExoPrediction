import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)

exo_df = pd.read_csv("data/exoplanets.csv")

plt.figure(figsize=(7, 4))
class_counts = exo_df["koi_disposition"].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.title("Class Distribution of Exoplanet Dispositions")
plt.xlabel("Disposition")
plt.ylabel("Count")
plt.savefig("plots/class_distribution_raw.png")
plt.close()


plt.figure(figsize=(25, 15))
sns.heatmap(exo_df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.savefig("plots/missing_values_heatmap.png")
plt.close()


numeric_cols = exo_df.select_dtypes(include=["int64", "float64"]).columns
corr_matrix = exo_df[numeric_cols].corr()

plt.figure(figsize=(25, 15))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numeric Features (Raw Dataset)")
plt.savefig("plots/correlation_heatmap_raw.png")
plt.close()


important_numeric = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_srad",
    "koi_steff"
]

for col in important_numeric:
    if col in exo_df.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(exo_df[col].dropna(), bins=30)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(f"plots/hist_{col}_raw.png")
        plt.close()


pairplot_features = [
    "koi_period",
    "koi_prad",
    "koi_depth",
    "koi_steff",
    "koi_srad",
    "koi_disposition"
]

pairplot_df = exo_df[pairplot_features].dropna()

sns.pairplot(pairplot_df, hue="koi_disposition", corner=True)
plt.savefig("plots/pairplot_selected_features_raw.png")
plt.close()
