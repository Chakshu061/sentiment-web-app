import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# Load sentiment model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# Title
st.title("📊 Sentiment Analysis Dashboard")
st.write("Analyze multiple texts, visualize sentiment distribution, and view detailed stats.")

# Session state for chat
if "texts" not in st.session_state:
    st.session_state.texts = []

# Input area
with st.container():
    user_input = st.text_area("💬 Enter multiple sentences (one per line):", height=150)

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Analyze Sentiment"):
            if user_input.strip():
                st.session_state.texts = [line.strip() for line in user_input.split("\n") if line.strip()]
            else:
                st.warning("Please enter some text.")
    with col2:
        if st.button("New Chat"):
            st.session_state.texts = []
            st.experimental_rerun()

# If texts exist, analyze them
if st.session_state.texts:
    st.subheader("🔍 Sentiment Results")
    results = sentiment_analyzer(st.session_state.texts, truncation=True)

    # Convert to DataFrame
    df = pd.DataFrame({
        "Text": st.session_state.texts,
        "Sentiment": [res["label"] for res in results],
        "Score": [round(res["score"], 2) for res in results]
    })

    # Display individual results
    st.dataframe(df, use_container_width=True)

    # Summary Stats
    st.subheader("📈 Summary Statistics")
    summary = df["Sentiment"].value_counts().reset_index()
    summary.columns = ["Sentiment", "Count"]

    col1, col2 = st.columns([1,1])
    with col1:
        fig1 = px.pie(summary, names="Sentiment", values="Count", title="Sentiment Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(summary, x="Sentiment", y="Count", title="Sentiment Counts", text="Count")
        st.plotly_chart(fig2, use_container_width=True)

    # Average score stats
    avg_scores = df.groupby("Sentiment")["Score"].mean().reset_index()
    st.write("### 📊 Average Confidence per Sentiment")
    st.bar_chart(avg_scores.set_index("Sentiment"))
