import feedparser
from textblob import TextBlob

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

class NewsAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

    def fetch_from_yahoo(self, ticker):
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:10]:
                articles.append({
                    'title': entry.title,
                    'description': entry.summary if 'summary' in entry else '',
                    'url': entry.link,
                    'published_at': entry.published if 'published' in entry else '',
                    'source': 'Yahoo Finance'
                })
            return articles
        except Exception as e:
            print(f"Error fetching Yahoo news: {e}")
            return []

    def analyze_sentiment(self, text):
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if self.vader:
                vader_scores = self.vader.polarity_scores(text)
                combined = (vader_scores['compound'] + polarity) / 2
            else:
                vader_scores = {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
                combined = polarity

            return {
                'vader': vader_scores,
                'textblob': {
                    'polarity': polarity,
                    'subjectivity': subjectivity
                },
                'combined_score': combined
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'vader': {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0},
                'textblob': {'polarity': 0, 'subjectivity': 0},
                'combined_score': 0
            }

    def analyze_articles(self, articles):
        for article in articles:
            text = f"{article['title']}. {article['description']}"
            article['sentiment'] = self.analyze_sentiment(text)
        return articles
