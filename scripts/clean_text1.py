import pandas as pd

df = pd.read_csv(r'D:\FakeNewsDetection\data\raw\fake.csv')  # Change to your actual filename

# Define fake/real logic—customize based on your dataset!
df['label'] = df['type'].apply(lambda x: 1 if x.lower() in ['bias', 'fake'] else 0)

# Combine title and text just like you did before
df['combined_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)

# Drop missing, blank, or duplicated combined_text or labels
df = df[['combined_text', 'label']].dropna()
df = df[df['combined_text'].str.strip() != '']
df = df.drop_duplicates()

df.to_csv('new_cleaned_dataset.csv', index=False)
