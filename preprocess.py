import re
import pandas as pd

def clean_text(text):
    if type(text) != str:
        return ""

    text = text.lower()

    # remove links
    text = re.sub(r"http\S+", " URLTOKEN ", text)
    text = re.sub(r"www\S+", " URLTOKEN ", text)

    # remove anything that isn't a letter or a space
    text = re.sub(r"[^a-z\s]", " ", text)

    # collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_dataset(csv_path):
    
    df = pd.read_csv(csv_path)

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    df["text"] = df["subject"] + " " + df["body"]

    df["text"] = df["text"].apply(clean_text)

    df = df[["text", "label"]]
    
    df = df[df["text"].str.len() > 0]

    df = df.drop_duplicates(subset="text")

    return df

if __name__ == "__main__":
    ceas = load_dataset("data/CEAS.csv")
    nazario = load_dataset("data/Nazario.csv")

    print("CEAS rows:", len(ceas))
    print(ceas["label"].value_counts())
    print()
    print("Nazario rows:", len(nazario))
    print(nazario["label"].value_counts())
    print()
    print("Example cleaned email:")
    print(ceas["text"].iloc[10][:200])
