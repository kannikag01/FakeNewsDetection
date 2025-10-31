import pandas as pd

# Load your current FakeNewsNet data (or whichever train CSV you want)
df1 = pd.read_csv(r"D:\FakeNewsDetection\data\processed\fakenewsnet_train.csv")
# Load your new cleaned dataset
df2 = pd.read_csv(r"D:\FakeNewsDetection\data\processed\new_cleaned_dataset.csv")

# Combine (stack) both DataFrames
df_combined = pd.concat([df1, df2], ignore_index=True)

# Save to a single CSV for BERT training
df_combined.to_csv(r"D:\FakeNewsDetection\data\processed\all_combined_train.csv", index=False)
print("Combined dataset shape:", df_combined.shape)
