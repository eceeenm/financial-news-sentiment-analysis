# Financial News Sentiment Analysis

This project classifies financial news into:

- Positive
- Neutral
- Negative

## Technologies

- Python
- Pandas
- NLTK
- Scikit-learn

## Model

- TF-IDF Vectorizer
- Logistic Regression

## Evaluation

- Classification Report
- Confusion Matrix

## Project Structure

```
financial-news-sentiment-analysis/
├── data/
│ └── all-data.csv
├── models/ # generated after running train.py (not tracked in git)
├── src/
│ ├── preprocess.py # loads and cleans the raw data
│ ├── train.py # trains the TF-IDF + Logistic Regression model
│ └── evaluate.py # evaluates the trained model
├── requirements.txt
└── README.md
```

## How to Run

1. Create a virtual environment and activate it:

python3 -m venv .venv
source .venv/bin/activate


2. Install dependencies:

pip install -r requirements.txt


3. Train the model:

cd src
python3 train.py


4. Evaluate the model:

python3 evaluate.py