import argparse
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLTK VADER lexicon dynamically
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

def analyze_sentiment(ticker):
    """
    Fetch recent news for a stock ticker from yFinance and calculate sentiment scores.
    """
    print(f"Fetching news sentiment for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        news = t.news
    except Exception as e:
        print(f"Failed to fetch news for {ticker}: {str(e)}")
        news = None

    if not news:
        print(f"No news articles found or failed to fetch for {ticker}.")
        return {
            "average_sentiment": 0.0,
            "sentiment_label": "Neutral",
            "articles": [],
            "stats": {"positive": 0, "negative": 0, "neutral": 0}
        }
    
    articles_data = []
    compound_sum = 0
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for article in news:
        title = article.get("title", "")
        publisher = article.get("publisher", "Unknown")
        link = article.get("link", "")
        pub_time = article.get("providerPublishTime", 0)
        
        # Analyze title sentiment
        scores = sia.polarity_scores(title)
        compound = scores["compound"]
        compound_sum += compound
        
        if compound >= 0.05:
            pos_count += 1
            label = "Positive"
        elif compound <= -0.05:
            neg_count += 1
            label = "Negative"
        else:
            neu_count += 1
            label = "Neutral"
            
        articles_data.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "pub_time": pub_time,
            "sentiment_score": compound,
            "sentiment_label": label
        })
        
    num_articles = len(articles_data)
    avg_compound = compound_sum / num_articles if num_articles > 0 else 0.0
    
    # Classify overall sentiment
    if avg_compound >= 0.05:
        overall_label = "Positive"
    elif avg_compound <= -0.05:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"
        
    result = {
        "average_sentiment": avg_compound,
        "sentiment_label": overall_label,
        "articles": articles_data,
        "stats": {
            "positive": pos_count,
            "negative": neg_count,
            "neutral": neu_count
        }
    }
    
    print(f"Processed {num_articles} news articles. Average compound score: {avg_compound:.4f} ({overall_label})")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and analyze news sentiment for a ticker")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    args = parser.parse_args()
    
    res = analyze_sentiment(args.ticker)
    for art in res["articles"][:3]:
        print(f"- [{art['sentiment_label']}] {art['title']} ({art['publisher']})")
