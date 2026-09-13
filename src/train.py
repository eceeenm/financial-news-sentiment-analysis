import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocess import load_and_clean

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model(data_path=None):
    df = load_and_clean(data_path)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean"])
    Y = df["sentiment"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, Y_train)

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(model, os.path.join(BASE_DIR, "models", "model.pkl"))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, "models", "vectorizer.pkl"))
    joblib.dump((X_test, Y_test), os.path.join(BASE_DIR, "models", "test_data.pkl"))

    return model, vectorizer, X_test, Y_test

if __name__ == "__main__":
    train_model()
    print("Model and vectorizer saved in models/ folder.")