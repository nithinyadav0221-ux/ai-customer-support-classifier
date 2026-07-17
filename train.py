from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Read the dataset
data = pd.read_csv("data/tickets.csv")

# Convert ticket text into numbers
vectorizer = TfidfVectorizer()
ticket_numbers = vectorizer.fit_transform(data["text"])

# Train category model
category_model = LogisticRegression()
category_model.fit(ticket_numbers, data["category"])

# Train priority model
priority_model = LogisticRegression()
priority_model.fit(ticket_numbers, data["priority"])

# Create folder for saved ML files
Path("models").mkdir(exist_ok=True)

# Save everything needed by the website
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(category_model, "models/category_model.joblib")
joblib.dump(priority_model, "models/priority_model.joblib")

print("Training complete. Category and priority models were saved.")