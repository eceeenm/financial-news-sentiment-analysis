import pandas as pd
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def load_and_clean(path):
    df = pd.read_csv(path, encoding="latin-1", header=None, names=["sentiment", "news"])
    df["word count"] = df["news"].str.split().str.len()
    df["clean"] = df["news"].str.lower()
    df["clean"] = df["clean"].str.replace("[^a-z\n]", " ", regex=True)
    df["clean"] = df["clean"].apply(
        lambda x: " ".join(word for word in x.split() if word not in stop_words)
    )
    return df

if __name__ == "__main__":
    df = load_and_clean("data/all-data.csv")
    print(df.head())
    print(df.groupby("sentiment")["word count"].mean())