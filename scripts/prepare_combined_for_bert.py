import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your cleaned news CSV
df = pd.read_csv(r"D:\FakeNewsDetection\data\processed\all_combined_train.csv")



df = df[['combined_text', 'label']].dropna()
df = df[df['combined_text'].str.strip() != ""]


# Train-test split (80% train, 20% test, stratify for balanced classes)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_df.to_csv(r"D:\FakeNewsDetection\data\processed\bert_train_combined.csv", index=False)
test_df.to_csv(r"D:\FakeNewsDetection\data\processed\bert_test_combined.csv", index=False)
print("New split shapes (train, test):", train_df.shape, test_df.shape)

