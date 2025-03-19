import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Generate Synthetic Dataset
def generate_review_sentiment_data(n=500):
    positive_words = ["excellent", "amazing", "great", "good", "awesome", "fantastic", "best", "love", "wonderful", "perfect"]
    negative_words = ["terrible", "awful", "worst", "bad", "horrible", "boring", "hate", "disappointing", "poor", "dull"]
    neutral_words = ["okay", "average", "fine", "decent", "mediocre", "not bad", "passable", "acceptable", "fair", "satisfactory"]
    
    reviews = []
    labels = []
    
    for _ in range(n//3):
        reviews.append(" ".join(random.choices(positive_words, k=random.randint(5, 15))))
        labels.append("positive")
        reviews.append(" ".join(random.choices(negative_words, k=random.randint(5, 15))))
        labels.append("negative")
        reviews.append(" ".join(random.choices(neutral_words, k=random.randint(5, 15))))
        labels.append("neutral")
    
    return pd.DataFrame({"review": reviews, "sentiment": labels})

# Create dataset
df = generate_review_sentiment_data(600)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

def predict_sentiment(reviews):
    input_vectors = vectorizer.transform(reviews)
    predictions = model.predict(input_vectors)
    return predictions

# Streamlit App
st.title("Movie Review Sentiment Analyzer")

option = st.selectbox("Choose an option:", ["Generate Random Reviews", "Input Custom Reviews", "Show Dataset"])

if option == "Generate Random Reviews":
    n = st.number_input("Enter the number of reviews to generate:", min_value=1, max_value=100, value=5)
    if st.button("Generate"):
        reviews = [" ".join(random.choices(positive_words + negative_words + neutral_words, k=random.randint(5, 15))) for _ in range(n)]
        predictions = predict_sentiment(reviews)
        for review, sentiment in zip(reviews, predictions):
            st.write(f"**Review:** {review}")
            st.write(f"**Predicted Sentiment:** {sentiment}")
            st.write("---")

elif option == "Input Custom Reviews":
    movie_name = st.text_input("Enter the movie name:")
    n = st.number_input("Enter the number of reviews you want to input:", min_value=1, max_value=50, value=5)
    user_reviews = []
    for i in range(n):
        user_reviews.append(st.text_input(f"Enter review {i+1}:"))
    if st.button("Analyze Reviews"):
        predictions = predict_sentiment(user_reviews)
        movie_df = pd.DataFrame({"review": user_reviews, "sentiment": predictions})
        st.write(movie_df)
        movie_filename = f"{movie_name.replace(' ', '_')}_reviews.csv"
        movie_df.to_csv(movie_filename, index=False)
        st.success(f"Dataset saved as {movie_filename}")

elif option == "Show Dataset":
    st.write(df.head(10))
