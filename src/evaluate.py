import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate():
    model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
    X_test, Y_test = joblib.load(os.path.join(BASE_DIR, "models", "test_data.pkl"))

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

if __name__ == "__main__":
    evaluate()