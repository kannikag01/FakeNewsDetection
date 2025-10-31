import pandas as pd
import re

# Load merged dataset
df = pd.read_csv(r"D:\FakeNewsDetection\data\processed\all_news_clean.csv")

# Remove rows with missing or empty text
df = df[df['text'].notnull() & (df['text'].str.strip() != "")]

# Optional: basic text clean (lowercase, remove non-alphanumerics except space)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Save to new file
df.to_csv(r"D:\FakeNewsDetection\data\processed\all_news_cleaned_text.csv", index=False)
print("Cleaned text data saved to data/processed/all_news_cleaned_text.csv")
