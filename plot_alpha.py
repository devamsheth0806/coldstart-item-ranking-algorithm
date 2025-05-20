import pandas as pd
import matplotlib.pyplot as plt

# Load alpha scores
alphas_df = pd.read_csv("optimized_alphas.csv", index_col=0)

# Check content
print("Alpha values (head):")
print(alphas_df.head())

# Line plot: alpha per user for each strategy
plt.figure(figsize=(10, 6))
for level in alphas_df.columns:
    plt.plot(alphas_df.index, alphas_df[level], label=f"{level.capitalize()} Strategy", marker='o')

plt.title("Optimized Alpha Values per Cold User")
plt.xlabel("Cold User ID")
plt.ylabel("Alpha Value")
plt.xticks(rotation=90)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("alpha_per_user.png")
plt.close()

# Boxplot: distribution of alpha across users per strategy
plt.figure(figsize=(8, 6))
alphas_df.boxplot()
plt.title("Distribution of Optimized Alpha Values per Strategy")
plt.ylabel("Alpha Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("alpha_boxplot.png")
plt.close()
