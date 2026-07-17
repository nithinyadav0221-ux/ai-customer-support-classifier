# AI Customer Support Ticket Classifier

An end-to-end machine learning and NLP project that analyzes customer-support tickets.

## Features

- Predicts ticket category: Billing, Account, Technical, or Delivery
- Predicts ticket priority: Low, Medium, or High
- Detects customer sentiment: Positive, Neutral, or Negative
- Suggests a support reply
- Saves analyzed tickets in a SQLite database
- Shows ticket-history analytics and charts
- Downloads ticket history as a CSV file

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- VADER Sentiment Analysis
- SQLite

## How to Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py