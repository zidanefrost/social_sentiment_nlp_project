import pandas as pd
from transformers import pipeline
import spacy
import sys
sample = int(sys.argv[1]) if len(sys.argv) > 1 else 500
df = pd.read_csv("tweets.csv").dropna(subset=["text"]).head(sample)

df = pd.read_csv("tweets.csv").dropna(subset=["text"]).head(500)

sent_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

df["sentiment"] = df["text"].apply(lambda t: sent_pipe(t)[0]["label"])
print(df[["text", "sentiment"]].head())
nlp = spacy.load("en_core_web_sm")
df["tokens"] = df["text"].apply(lambda t: [tok.text for tok in nlp(t)])
df.to_csv("tweets_with_sentiment.csv", index=False)
print("Saved tweets_with_sentiment.csv")
