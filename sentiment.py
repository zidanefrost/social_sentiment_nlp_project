import pandas as pd
from transformers import pipeline

df = pd.read_csv("tweets.csv").dropna(subset=["text"]).head(500)

sent_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

df["sentiment"] = df["text"].apply(lambda t: sent_pipe(t)[0]["label"])
print(df[["text", "sentiment"]].head())
