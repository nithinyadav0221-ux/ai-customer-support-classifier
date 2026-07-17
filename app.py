import sqlite3

import joblib
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load trained ML files
vectorizer = joblib.load("models/vectorizer.joblib")
category_model = joblib.load("models/category_model.joblib")
priority_model = joblib.load("models/priority_model.joblib")

# Create NLP sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()


def create_database():
    connection = sqlite3.connect("tickets.db")
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            ticket TEXT,
            category TEXT,
            priority TEXT,
            confidence REAL
        )
    """)

    connection.commit()
    connection.close()


def save_ticket(ticket, category, priority, confidence):
    connection = sqlite3.connect("tickets.db")
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO ticket_history (ticket, category, priority, confidence)
        VALUES (?, ?, ?, ?)
    """, (ticket, category, priority, confidence))

    connection.commit()
    connection.close()


create_database()

st.title("AI Customer Support Ticket Classifier")
st.write("Enter a customer complaint to predict category, priority, and sentiment.")

ticket = st.text_area("Describe the customer issue")

if st.button("Analyze"):
    if not ticket.strip():
        st.warning("Please type an issue first.")
    else:
        # Convert entered ticket text into numbers
        ticket_numbers = vectorizer.transform([ticket])

        # ML category and priority prediction
        category = category_model.predict(ticket_numbers)[0]
        priority = priority_model.predict(ticket_numbers)[0]

        probabilities = category_model.predict_proba(ticket_numbers)[0]
        confidence = max(probabilities) * 100

        # NLP sentiment analysis
        sentiment_score = sentiment_analyzer.polarity_scores(ticket)["compound"]

        if sentiment_score >= 0.05:
            sentiment = "Positive"
        elif sentiment_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Save analysis in database
        save_ticket(ticket, category, priority, confidence)

        st.success(f"Category: {category}")
        st.warning(f"Priority: {priority}")
        st.info(f"Customer sentiment: {sentiment}")
        st.info(f"Category confidence: {confidence:.1f}%")

        replies = {
            "Billing": "We are sorry for the payment issue. Our billing team will review the transaction and update you soon.",
            "Account": "We are sorry you cannot access your account. Please try resetting your password, and our account team can assist further.",
            "Technical": "We are sorry you are facing a technical problem. Please share your device and app version with our technical team.",
            "Delivery": "We are checking your delivery status. Our delivery team will share an update shortly."
        }

        st.subheader("Suggested support reply")
        st.write(replies.get(category, "A support agent will review your issue shortly."))

st.divider()
st.subheader("Recent ticket history")

connection = sqlite3.connect("tickets.db")

history = pd.read_sql_query("""
    SELECT created_at, ticket, category, priority, ROUND(confidence, 1) AS confidence_percent
    FROM ticket_history
    ORDER BY id DESC
    LIMIT 10
""", connection)

connection.close()

st.dataframe(history, use_container_width=True)
if not history.empty:
    csv_data = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download ticket history as CSV",
        data=csv_data,
        file_name="ticket_history.csv",
        mime="text/csv"
    )
if not history.empty:
    csv_data = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download ticket history as CSV",
        data=csv_data,
        file_name="ticket_history.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("Ticket Analytics")

if history.empty:
    st.info("Analyze some tickets to see analytics.")
else:
    total_tickets = len(history)
    high_priority_count = len(history[history["priority"] == "High"])

    first_column, second_column = st.columns(2)
    first_column.metric("Tickets analyzed", total_tickets)
    second_column.metric("High-priority tickets", high_priority_count)

    st.subheader("Tickets by category")
    category_counts = history["category"].value_counts()
    st.bar_chart(category_counts)

    st.subheader("Tickets by priority")
    priority_counts = history["priority"].value_counts()
    st.bar_chart(priority_counts)
    