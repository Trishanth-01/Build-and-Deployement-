

import requests
from transformers import pipeline
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Your TMDB API Key
TMDB_API_KEY = "c6eb16d5e85900a65b01f6d44cfe2a6c"

# Sentiment pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


def preprocess_text(text):
    return ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])


# Fetch Movie Plot from TMDB
def get_movie_plot(movie_name):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    r = requests.get(search_url).json()

    if not r["results"]:
        return "Error: Movie not found."

    movie_id = r["results"][0]["id"]

    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    details = requests.get(details_url).json()

    return details.get("overview", "Plot not available.")


# Sentiment classification
def get_sentiment(review):
    sentiment = sentiment_analyzer(review)[0]
    return sentiment["label"], sentiment["score"]


# Plot sentiment result
def plot_sentiment(label, score):
    colors = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}
    height = score if label == "POSITIVE" else (1-score if label == "NEGATIVE" else 0.5)
    plt.bar(label, height, color=colors.get(label, "gray"))
    plt.title(f"Sentiment: {label} (Score: {score:.2f})")
    plt.ylim(0, 1)
    plt.show()


# Movie Assistant
def movie_assistant():
    print("🎬 Welcome to the Movie Assistant!")

    movie_name = input("Enter a movie name: ").strip()
    plot = get_movie_plot(movie_name)

    if "Error" in plot:
        print(plot)
        return

    print("\nMovie Plot:")
    print(plot)

    review = input("\nEnter your review: ")
    label, score = get_sentiment(review)
    print(f"Sentiment: {label} ({score:.2f})")
    plot_sentiment(label, score)


# Run
movie_assistant()
