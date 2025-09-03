# app.py
import os
# --- Force CPU / avoid MPS meta-tensor issues on macOS Apple silicon ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_DISABLE"] = "1"

import streamlit as st
from transformers import pipeline, AutoTokenizer
import pandas as pd
import requests
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# ---------------------------
# Config / Constants
# ---------------------------
DEFAULT_MODEL = "ProsusAI/finbert"   # FinBERT for financial sentiment
NEWSAPI_URL = "https://newsapi.org/v2/everything"
# Keep a small safety margin (FinBERT is ok for headlines; chunking included for long text)
SAFE_TOKEN_LIMIT = 480

# ---------------------------
# Cache & Load Model
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_finbert(model_name=DEFAULT_MODEL):
    # Return both tokenizer and pipeline (analyzer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    analyzer = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer, device=-1)
    return analyzer, tokenizer

analyzer, tokenizer = load_finbert()

# ---------------------------
# Helper functions
# ---------------------------
def fetch_news_newsapi(query, from_dt=None, to_dt=None, api_key=None, page_size=100, language="en"):
    """
    Fetch headlines from NewsAPI.org for `query`. Requires an API key.
    Returns list of dicts: [{'title':..., 'description':..., 'publishedAt':...}, ...]
    """
    if not api_key:
        return []
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "sortBy": "relevancy"
    }
    if from_dt:
        params["from"] = from_dt.isoformat()
    if to_dt:
        params["to"] = to_dt.isoformat()
    headers = {"Authorization": api_key}
    resp = requests.get(NEWSAPI_URL, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", [])
    results = []
    for a in articles:
        results.append({
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "content": (a.get("content") or ""),
            "publishedAt": a.get("publishedAt")
        })
    return results

def chunk_text_by_tokens(text, tokenizer, max_tokens=SAFE_TOKEN_LIMIT):
    # Encode whole text into token ids and slice tokens to safe chunks
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        slice_ids = token_ids[i:i+max_tokens]
        chunk_text = tokenizer.decode(slice_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
    return chunks

def analyze_headlines(headlines, analyzer, tokenizer, batch_size=16):
    """
    headlines: list of strings (title / description combined)
    returns: DataFrame with columns: text, label, score
    """
    safe_texts = []
    original_index = []
    # Expand very long headlines into chunks (rare) to avoid token overflow
    for idx, h in enumerate(headlines):
        if not h:
            continue
        encoded_len = len(tokenizer.encode(h, add_special_tokens=False))
        if encoded_len > SAFE_TOKEN_LIMIT:
            # chunk it
            chs = chunk_text_by_tokens(h, tokenizer, SAFE_TOKEN_LIMIT)
            for c in chs:
                safe_texts.append(c)
                original_index.append(idx)
        else:
            safe_texts.append(h)
            original_index.append(idx)

    # Run pipeline in batches
    results = analyzer(safe_texts, batch_size=batch_size, truncation=True, max_length=512)

    # aggregate back to original headline index: we'll average scores per original headline (if chunked)
    rows = []
    for orig_idx in range(max(original_index)+1 if original_index else 0):
        rows.append({"text": None, "label_scores": [], "labels": []})

    for txt, res, oidx in zip(safe_texts, results, original_index):
        # normalize label names to standard: positive/neutral/negative (FinBERT uses 'positive','negative','neutral')
        label = res.get("label")
        if isinstance(label, str):
            label_norm = label.lower()
        else:
            label_norm = str(label).lower()
        rows[oidx]["text"] = headlines[oidx]
        rows[oidx]["label_scores"].append((label_norm, float(res.get("score", 0.0))))
        rows[oidx]["labels"].append(label_norm)

    # Build DataFrame: compute aggregated label by summed score
    output = []
    for r in rows:
        if r["text"] is None:
            continue
        # sum scores per label
        score_map = {}
        for lab, sc in r["label_scores"]:
            score_map[lab] = score_map.get(lab, 0.0) + sc
        # choose label with highest summed score
        if score_map:
            agg_label = max(score_map.items(), key=lambda x: x[1])[0]
            agg_score = score_map[agg_label] / sum(score_map.values())
        else:
            agg_label = "neutral"
            agg_score = 0.0
        output.append({
            "text": r["text"],
            "label": agg_label,
            "score": round(agg_score, 4)
        })
    df = pd.DataFrame(output)
    return df

def rolling_sentiment_timeseries(df, date_col="publishedAt", label_col="label", score_col="score", freq="D", window=3):
    """
    df must contain publishedAt timestamps (ISO string) - returns timeseries DataFrame
    """
    if date_col not in df.columns:
        df[date_col] = pd.Timestamp.now()
    # normalize
    df = df.copy()
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce').fillna(pd.Timestamp.now())
    # create numeric sentiment score: positive=1, neutral=0, negative=-1 scaled by score
    def numeric(r):
        if r[label_col] == "positive":
            return r[score_col] * 1.0
        elif r[label_col] == "negative":
            return -1.0 * r[score_col]
        else:
            return 0.0
    df['numeric'] = df.apply(numeric, axis=1)
    ts = df.set_index('publishedAt').resample(freq).agg({
        'numeric': ['mean', 'count']
    })
    ts.columns = ['sentiment_mean', 'count']
    ts['rolling'] = ts['sentiment_mean'].rolling(window=window, min_periods=1).mean()
    ts = ts.reset_index()
    return ts

def make_wordcloud(texts, max_words=100):
    joined = " ".join([t for t in texts if isinstance(t, str)])
    wc = WordCloud(width=800, height=450, background_color="white", max_words=max_words).generate(joined)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide")
st.title("📈 Financial Sentiment Dashboard — FinBERT")

# Sidebar controls
st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Demo (sample headlines)", "Live (NewsAPI)", "Manual (paste headlines)"])
model_display = st.sidebar.text_input("Model (editable)", value=DEFAULT_MODEL)
if model_display != DEFAULT_MODEL:
    # reload model if user typed different
    try:
        analyzer, tokenizer = load_finbert(model_display)
        st.sidebar.success(f"Loaded model {model_display}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

if mode == "Live (NewsAPI)":
    newsapi_key = st.sidebar.text_input("NewsAPI Key (get from newsapi.org)", type="password")
    query = st.sidebar.text_input("Query / Company (e.g., Tesla OR TSLA OR \"Tesla Inc\")", value="Tesla")
    days_back = st.sidebar.slider("Days back", 1, 30, 7)
    from_dt = dt.datetime.utcnow() - dt.timedelta(days=days_back)
    to_dt = dt.datetime.utcnow()
else:
    newsapi_key = None
    query = None
    from_dt = None
    to_dt = None

st.sidebar.markdown("---")
threshold_alert = st.sidebar.slider("Alert threshold (rolling sentiment)", -1.0, 1.0, -0.3, 0.05)
rolling_window = st.sidebar.slider("Rolling window (days)", 1, 14, 3)
st.sidebar.markdown("Made with FinBERT — `ProsusAI/finbert`")

# Main interaction
if mode == "Demo (sample headlines)":
    st.markdown("### Demo headlines (financial news examples)")
    sample_headlines = [
        "Tesla shares jump after record quarterly deliveries",
        "Apple warns iPhone production may be delayed due to supply chain issues",
        "Amazon beats expectations as cloud revenue grows faster than predicted",
        "Google faces antitrust scrutiny in the EU, shares fall",
        "Microsoft invests $10B into new AI research center",
        "Markets slide as inflation fears resurface"
    ]
    df_headlines_meta = pd.DataFrame({
        "text": [h for h in sample_headlines],
        "publishedAt": [pd.Timestamp.now().isoformat()] * len(sample_headlines)
    })
    st.info("Demo mode: using built-in sample headlines.")
    run_button = st.button("Analyze Demo Headlines")
elif mode == "Live (NewsAPI)":
    st.markdown("### Live news from NewsAPI.org")
    st.write(f"Query: **{query}**  ·  Date range: **{from_dt.date()} → {to_dt.date()}**")
    run_button = st.button("Fetch & Analyze Live News")
else:
    st.markdown("### Paste headlines (one per line). Optionally include timestamp after a '||' (ISO format).")
    manual_input = st.text_area("Paste headlines here:", height=200)
    run_button = st.button("Analyze Pasted Headlines")

# Run pipeline depending on mode
results_df = pd.DataFrame()
if run_button:
    with st.spinner("Fetching and analyzing..."):
        try:
            if mode == "Demo (sample headlines)":
                headlines = df_headlines_meta['text'].tolist()
                meta_dates = df_headlines_meta['publishedAt'].tolist()
            elif mode == "Live (NewsAPI)":
                if not newsapi_key:
                    st.error("Please provide NewsAPI key in the sidebar.")
                    st.stop()
                fetched = fetch_news_newsapi(query, from_dt=from_dt, to_dt=to_dt, api_key=newsapi_key, page_size=100)
                if not fetched:
                    st.warning("No articles found for given query / date range.")
                    st.stop()
                headlines = [f"{a['title']} — {a['description'] or ''}".strip() for a in fetched]
                meta_dates = [a['publishedAt'] for a in fetched]
            else:  # manual
                raw_lines = [l.strip() for l in manual_input.split("\n") if l.strip()]
                headlines = []
                meta_dates = []
                for ln in raw_lines:
                    if "||" in ln:
                        parts = ln.split("||", 1)
                        headlines.append(parts[0].strip())
                        meta_dates.append(parts[1].strip())
                    else:
                        headlines.append(ln)
                        meta_dates.append(pd.Timestamp.now().isoformat())
            # Analyze
            df = analyze_headlines(headlines, analyzer, tokenizer)
            df['publishedAt'] = meta_dates[:len(df)]
            results_df = df.copy()
            st.success(f"Analyzed {len(df)} headlines")
        except Exception as e:
            st.exception(e)

# If we have results, show dashboard
if not results_df.empty:
    st.markdown("## Results")
    # Show top-level table
    st.dataframe(results_df[['publishedAt','text','label','score']].rename(columns={'publishedAt':'date','text':'headline'}), use_container_width=True)

    # Timeline / timeseries
    ts = rolling_sentiment_timeseries(results_df)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts['publishedAt'], y=ts['count'], name='Article count', marker_color='lightgrey', yaxis='y2', opacity=0.6))
    fig.add_trace(go.Scatter(x=ts['publishedAt'], y=ts['sentiment_mean'], name='Daily mean sentiment', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=ts['publishedAt'], y=ts['rolling'], name=f'{rolling_window}-day rolling mean', line=dict(width=3)))
    fig.update_layout(
        title="Sentiment over time",
        xaxis_title="Date",
        yaxis_title="Sentiment score (-1 to 1)",
        yaxis=dict(range=[-1,1]),
        yaxis2=dict(overlaying='y', side='right', title='Article count'),
        legend=dict(orientation='h')
    )
    st.plotly_chart(fig, use_container_width=True)

    # compute distribution
    dist = results_df['label'].value_counts().rename_axis('label').reset_index(name='count')

    # Charts in two columns
    c1, c2 = st.columns([2,1])
    with c1:
        fig2 = px.pie(dist, names='label', values='count', title='Overall Sentiment Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.bar(dist, x='label', y='count', title='Counts by Sentiment', text='count')
        st.plotly_chart(fig3, use_container_width=True)

    # Top positive and negative headlines
    st.markdown("### Top headlines")
    pos = results_df[results_df['label']=='positive'].sort_values('score', ascending=False).head(5)
    neg = results_df[results_df['label']=='negative'].sort_values('score', ascending=False).head(5)
    neu = results_df[results_df['label']=='neutral'].sort_values('score', ascending=False).head(5)

    st.subheader("Top Positive Headlines")
    for _, row in pos.iterrows():
        st.write(f"- ({row['score']:.2f}) {row['text']}")

    st.subheader("Top Negative Headlines")
    for _, row in neg.iterrows():
        st.write(f"- ({row['score']:.2f}) {row['text']}")

    # Wordcloud
    st.markdown("### Word Cloud (all headlines)")
    wc_buf = make_wordcloud(results_df['text'].tolist())
    st.image(wc_buf)

    # Alerts based on rolling sentiment
    latest_rolling = float(ts['rolling'].iloc[-1]) if not ts.empty else 0.0
    st.markdown("### Alerts")
    if latest_rolling <= threshold_alert:
        st.error(f"⚠️ Rolling sentiment is {latest_rolling:.2f} which is below threshold {threshold_alert:.2f}. Consider downside risk.")
    else:
        st.success(f"Rolling sentiment is {latest_rolling:.2f} — above threshold.")

    # Download CSV
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download results CSV", csv, file_name="fin_sentiment_results.csv", mime="text/csv")

    # Small explanation / interpretation
    st.markdown("---")
    st.markdown("**Interpretation tips**:")
    st.markdown("- A sustained negative rolling sentiment can indicate increasing market risk for the queried topic/stock.")
    st.markdown("- Use sentiment with other signals (price, volume, fundamentals). This is a signal, not a trading rule.")
    st.markdown("- You can schedule this script to run periodically (Cloud Run / Cron) and store results for longer historical analysis.")
