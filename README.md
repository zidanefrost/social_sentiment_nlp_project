# Social Sentiment Analysis Pipeline 🧠💬

> Small weekend NLP project analyzing sentiment of tweets using HuggingFace Transformers and SpaCy.  
> Designed to experiment with LLM text classification and tokenization workflows on social media data.

---

## 🔍 What It Does

- ✅ Loads ~500 tweets from a `.csv`
- ✅ Tokenizes using **SpaCy**
- ✅ Analyzes sentiment using **DistilBERT**
- ✅ Exports results to `tweets_with_sentiment.csv`
- ✅ Can be run with optional sample size argument

---

## 🛠️ Tech Stack

| Library         | Purpose                     |
|------------------|-----------------------------|
| `pandas`         | Data loading & export        |
| `transformers`   | Sentiment pipeline (DistilBERT) |
| `SpaCy`          | Tokenization                |
| `torch`          | Model backend (required)     |

---

## 📦 Setup & Run

```bash
git clone https://github.com/yourhandle/social_sentiment_nlp_project.git
cd social_sentiment_nlp_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python sentiment.py 500
