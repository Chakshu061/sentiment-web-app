# Sentiment Analysis Dashboard

📊 A web application built with Streamlit that performs sentiment analysis on multiple texts, visualizes sentiment distribution, and provides detailed statistics.

## Features

- Analyze sentiment for multiple sentences at once.
- View sentiment results in a tabular format.
- Visualize sentiment distribution using pie and bar charts.
- Display average confidence scores for each sentiment.

## Demo

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Sentiment+Analysis+Dashboard)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sentiment-web-app.git
   cd sentiment-web-app
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.

## File Structure

```
sentiment-web-app/
├── app.py                  # Main Streamlit app
├── sentiment_analysis.py   # Standalone sentiment analysis script
└── requirements.txt        # Python dependencies
```

## Usage

### Sentiment Analysis Dashboard

1. Enter multiple sentences in the text area (one per line).
2. Click **Analyze Sentiment** to process the input.
3. View the results in the following sections:
   - **Sentiment Results**: A table showing the sentiment and confidence score for each sentence.
   - **Summary Statistics**: Pie and bar charts visualizing the sentiment distribution.
   - **Average Confidence per Sentiment**: A bar chart showing the average confidence score for each sentiment.

4. Click **New Chat** to reset the input and start over.

### Standalone Script

You can also use the `sentiment_analysis.py` script to analyze a single text:

```bash
python sentiment_analysis.py
```

## Dependencies

- Python 3.7+
- Streamlit
- Transformers
- Pandas
- Plotly

Install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Example

### Input

```
I love this product!
This is the worst experience I've ever had.
The service was okay, nothing special.
```

### Output

#### Sentiment Results

| Text                                      | Sentiment | Score |
|-------------------------------------------|-----------|-------|
| I love this product!                      | POSITIVE  | 0.99  |
| This is the worst experience I've ever had. | NEGATIVE  | 0.98  |
| The service was okay, nothing special.    | NEUTRAL   | 0.75  |

#### Summary Statistics

- **Pie Chart**: Sentiment distribution.
- **Bar Chart**: Sentiment counts.

#### Average Confidence per Sentiment

- Bar chart showing average confidence scores.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the sentiment analysis pipeline.
- [Streamlit](https://streamlit.io/) for the web app framework.
- [Plotly](https://plotly.com/) for data visualization.

## Author

- [Your Name](https://github.com/your-username)
