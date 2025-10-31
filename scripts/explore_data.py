import pandas as pd
import scipy.io

# Set your data root for convenience
data_root = r"D:\FakeNewsDetection\data\raw\fakenewsnet"

# List of files to explore (add/remove paths as needed)
csv_files = [
    "BuzzFeed_fake_news_content.csv",
    "BuzzFeed_real_news_content.csv",
    "BuzzFeedNews.txt",
    "BuzzFeedNewsUser.txt",
    "BuzzFeedUser.txt",
    "BuzzFeedUserUser.txt",
    "PolitiFact_fake_news_content.csv",
    "PolitiFact_real_news_content.csv",
    "PolitiFactNews.txt",
    "PolitiFactNewsUser.txt",
    "PolitiFactUser.txt",
    "PolitiFactUserUser.txt"
]

mat_files = [
    "BuzzFeedUserFeature.mat",
    "PolitiFactUserFeature.mat"
]

print("=== CSV/TXT FILES ===")
for fname in csv_files:
    fpath = f"{data_root}\\{fname}"
    try:
        # Try comma, else tab separator
        try:
            df = pd.read_csv(fpath, encoding="utf-8")
        except:
            df = pd.read_csv(fpath, sep="\t", encoding="utf-8")
        print(f"\nFile: {fname}")
        print(df.head())
        print("Columns:", df.columns.values)
        print(df.info())
    except Exception as e:
        print(f"Could not read {fname}: {e}")

print("\n=== MAT FILES ===")
for fname in mat_files:
    fpath = f"{data_root}\\{fname}"
    try:
        mat_data = scipy.io.loadmat(fpath)
        print(f"\nFile: {fname}")
        print("Keys:", mat_data.keys())
        # Show a summary of one variable, if present
        for key in mat_data:
            if not key.startswith("__"):
                print(f"{key} shape/type: {type(mat_data[key])} {getattr(mat_data[key], 'shape', '')}")
                break
    except Exception as e:
        print(f"Could not read {fname}: {e}")
