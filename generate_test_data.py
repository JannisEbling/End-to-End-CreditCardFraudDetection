import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read the original dataset
df = pd.read_csv("data/creditcard_2023.csv")

# Take 5000 random samples
sampled_df = df.sample(n=5000, random_state=42)

# Save to CSV
output_path = "data/test_creditcard_data.csv"
sampled_df.to_csv(output_path, index=False)
print(
    f"Sampled {len(sampled_df)} cases from creditcard_2023 dataset and saved to {output_path}"
)
