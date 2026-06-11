import pandas as pd
import numpy as np
import re
import nltk
nltk.download("stopwords", quiet = True)
from nltk.corpus import stopwords
stop_words =set(stopwords.words("english"))
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

df = pd.read_csv("all-data.csv", encoding= "latin-1", header=None, names= ["sentiment", "news"])
df.head()
df.shape
df.value_counts("sentiment")
df.groupby("sentiment").count()
df["word count"] = df ["news"].str.split().str.len()
df.groupby("sentiment")["word count"].mean()

df["clean"] = df["news"].str.lower()
df["clean"] = df["clean"].str.replace("[^a-z\n]", " ", regex=True) 
df["clean"] = df["clean"].apply(lambda x:" ".join(word for word in x.split() if word not in stop_words)) 
df.head()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean"])
print(X.shape)

Y = df["sentiment"] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(class_weight = "balanced") 
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(Y_pred[:10])

print(classification_report(Y_test, Y_pred)) 

print(confusion_matrix(Y_test, Y_pred))