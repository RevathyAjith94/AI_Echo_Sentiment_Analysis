import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Echo", layout="wide")

st.title("🤖 AI Echo: Smart Sentiment Analysis Dashboard")
st.write("Analyze ChatGPT user reviews using NLP & Machine Learning")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/cleaned_reviews.csv")

# -----------------------------
# CREATE SENTIMENT COLUMN (FIX FOR KeyError)
# -----------------------------
def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["rating"].apply(label_sentiment)

# -----------------------------
# LOAD MODEL & VECTORIZER
# -----------------------------
model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔍 Filters")

platform_filter = st.sidebar.multiselect(
    "Select Platform",
    options=df["platform"].unique(),
    default=df["platform"].unique()
)

version_filter = st.sidebar.multiselect(
    "Select ChatGPT Version",
    options=df["version"].unique(),
    default=df["version"].unique()
)

filtered_df = df[
    (df["platform"].isin(platform_filter)) &
    (df["version"].isin(version_filter))
].copy()   # 👈 IMPORTANT FIX

# -----------------------------
# KPI METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", len(filtered_df))
col2.metric("Average Rating", round(filtered_df["rating"].mean(), 2))
col3.metric(
    "Verified Users (%)",
    round((filtered_df["verified_purchase"] == "Yes").mean() * 100, 1)
)

# -----------------------------
# SENTIMENT DISTRIBUTION
# -----------------------------
st.subheader("📊 Sentiment Distribution")

sentiment_counts = filtered_df["sentiment"].value_counts()

fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
st.pyplot(fig)

# -----------------------------
# RATING DISTRIBUTION
# -----------------------------
st.subheader("⭐ Rating Distribution")

fig, ax = plt.subplots()
filtered_df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
st.pyplot(fig)

# -----------------------------
# SENTIMENT VS RATING
# -----------------------------
st.subheader("🔁 Sentiment vs Rating")

fig, ax = plt.subplots()
filtered_df.groupby("rating")["sentiment"].value_counts().unstack().plot(ax=ax)
st.pyplot(fig)

# -----------------------------
# WORD CLOUDS
# -----------------------------
st.subheader("☁️ Word Clouds")

col1, col2 = st.columns(2)

positive_text = " ".join(
    filtered_df[filtered_df["sentiment"] == "Positive"]["clean_review"]
)
negative_text = " ".join(
    filtered_df[filtered_df["sentiment"] == "Negative"]["clean_review"]
)

with col1:
    st.write("Positive Reviews")
    if positive_text.strip():
        wc = WordCloud(background_color="white").generate(positive_text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No positive reviews for selected filters")

with col2:
    st.write("Negative Reviews")
    if negative_text.strip():
        wc = WordCloud(background_color="black", colormap="Reds").generate(negative_text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No negative reviews for selected filters")

# -----------------------------
# TIME SERIES ANALYSIS (SAFE DATE PARSING)
# -----------------------------
st.subheader("📈 Average Rating Over Time")

filtered_df["date"] = pd.to_datetime(
    filtered_df["date"], errors="coerce"
)
filtered_df = filtered_df.dropna(subset=["date"])

time_df = filtered_df.groupby(
    filtered_df["date"].dt.to_period("M")
)["rating"].mean()

fig, ax = plt.subplots()
time_df.plot(ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Average Rating")
st.pyplot(fig)

# -----------------------------
# SENTIMENT PREDICTOR
# -----------------------------
st.subheader("🧠 Predict Sentiment")

user_input = st.text_area("Enter a user review")

if st.button("Predict Sentiment"):
    if user_input.strip():
        vectorized = vectorizer.transform([user_input.lower()])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text")
