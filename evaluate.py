import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/tickets.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["category"],
    test_size=0.33,
    random_state=42,
    stratify=data["category"]
)

vectorizer = TfidfVectorizer()
X_train_numbers = vectorizer.fit_transform(X_train)
X_test_numbers = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_numbers, y_train)

predictions = model.predict(X_test_numbers)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print()
print(classification_report(y_test, predictions, zero_division=0))