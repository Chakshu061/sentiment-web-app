from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Sample text
text = "I love learning AI with Hugging Face, it's amazing!"

# Analyze sentiment
result = sentiment_analyzer(text)
print(result)
