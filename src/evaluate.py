import joblib
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    model = joblib.load("models/model.pkl")
    X_test, Y_test = joblib.load("models/test_data.pkl")

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

if __name__ == "__main__":
    evaluate()