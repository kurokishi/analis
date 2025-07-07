import feedparser
from textblob import TextBlob
import logging
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self, language='en'):
        """
        Inisialisasi analisis berita
        
        Args:
            language: Bahasa untuk analisis sentimen ('en' atau 'id')
        """
        self.language = language
        self.sources = {
            'yahoo': 'Yahoo Finance',
            'idx': 'Bursa Efek Indonesia',
            'kontan': 'Kontan',
            'google': 'Google News'
        }
        
        # Inisialisasi analyzer sentimen
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment tidak terinstal, hanya menggunakan TextBlob")
            self.vader = None

    def fetch_news(self, ticker, max_articles=15):
        """
        Mengambil berita dari berbagai sumber untuk ticker tertentu
        
        Args:
            ticker: Simbol saham (contoh: 'BBCA.JK')
            max_articles: Jumlah maksimal artikel per sumber
            
        Returns:
            List artikel dari berbagai sumber
        """
        articles = []
        
        # Yahoo Finance (untuk saham internasional dan Indonesia)
        articles += self.fetch_from_yahoo(ticker, max_articles)
        
        # IDX (Bursa Efek Indonesia)
        if ticker.endswith('.JK'):
            articles += self.fetch_from_idx(ticker.replace('.JK', ''), max_articles)
            
        # Kontan (situs berita ekonomi Indonesia)
        if ticker.endswith('.JK'):
            articles += self.fetch_from_kontan(ticker.replace('.JK', ''), max_articles)
            
        # Google News (pencarian umum)
        articles += self.fetch_from_google(ticker, max_articles)
        
        return articles[:max_articles * 2]  # Batasi total artikel

    def fetch_from_yahoo(self, ticker, max_articles=10):
        """Mengambil berita dari Yahoo Finance"""
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:max_articles]:
                articles.append({
                    'title': entry.title,
                    'description': entry.summary if 'summary' in entry else '',
                    'url': entry.link,
                    'published_at': entry.published if 'published' in entry else '',
                    'source': 'Yahoo Finance',
                    'ticker': ticker
                })
            return articles
        except Exception as e:
            logger.error(f"Error mengambil berita Yahoo: {e}")
            return []

    def fetch_from_idx(self, ticker, max_articles=10):
        """Mengambil berita dari situs Bursa Efek Indonesia (IDX)"""
        try:
            url = f"https://www.idx.co.id/umbraco/Surface/StockNews/GetStockNews?code={ticker}&category=all"
            response = requests.get(url)
            news_data = response.json()
            
            articles = []
            for item in news_data[:max_articles]:
                # Format tanggal dari timestamp
                pub_date = datetime.fromtimestamp(item['Date']/1000).strftime('%a, %d %b %Y %H:%M:%S %z')
                
                articles.append({
                    'title': item['Title'],
                    'description': item['Summary'],
                    'url': f"https://www.idx.co.id{item['Url']}",
                    'published_at': pub_date,
                    'source': 'Bursa Efek Indonesia',
                    'ticker': f"{ticker}.JK"
                })
            return articles
        except Exception as e:
            logger.error(f"Error mengambil berita IDX: {e}")
            return []

    def fetch_from_kontan(self, ticker, max_articles=10):
        """Mengambil berita dari Kontan.co.id"""
        try:
            url = f"https://search.kontan.co.id/api/search?query={ticker}&sort=date&type=article"
            response = requests.get(url)
            news_data = response.json()
            
            articles = []
            for item in news_data.get('data', [])[:max_articles]:
                # Ekstrak konten singkat
                soup = BeautifulSoup(item['content'], 'html.parser')
                description = soup.get_text()[:200] + "..." if soup.get_text() else ""
                
                articles.append({
                    'title': item['title'],
                    'description': description,
                    'url': item['url'],
                    'published_at': item['date'],
                    'source': 'Kontan',
                    'ticker': f"{ticker}.JK"
                })
            return articles
        except Exception as e:
            logger.error(f"Error mengambil berita Kontan: {e}")
            return []

    def fetch_from_google(self, ticker, max_articles=10):
        """Mengambil berita dari Google News"""
        try:
            # Untuk saham Indonesia, gunakan kode tanpa .JK
            query = ticker.replace('.JK', '') if ticker.endswith('.JK') else ticker
            url = f"https://news.google.com/rss/search?q={query}+saham&hl=id&gl=ID&ceid=ID:id"
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                articles.append({
                    'title': entry.title,
                    'description': entry.description if 'description' in entry else '',
                    'url': entry.link,
                    'published_at': entry.published if 'published' in entry else '',
                    'source': 'Google News',
                    'ticker': ticker
                })
            return articles
        except Exception as e:
            logger.error(f"Error mengambil berita Google: {e}")
            return []

    def analyze_sentiment(self, text):
        """Menganalisis sentimen teks dengan dukungan bahasa Indonesia"""
        try:
            # Pra-pemrosesan teks
            cleaned_text = self.clean_text(text)
            
            # Analisis dengan TextBlob (terjemahkan jika bahasa Indonesia)
            if self.language == 'id':
                translated = self.translate_text(cleaned_text)
                blob = TextBlob(translated)
            else:
                blob = TextBlob(cleaned_text)
                
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Analisis dengan VADER (hanya untuk bahasa Inggris)
            vader_scores = {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
            if self.vader and self.language == 'en':
                vader_scores = self.vader.polarity_scores(cleaned_text)
                combined = (vader_scores['compound'] + polarity) / 2
            else:
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
            logger.error(f"Error analisis sentimen: {e}")
            return {
                'vader': {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0},
                'textblob': {'polarity': 0, 'subjectivity': 0},
                'combined_score': 0
            }

    def clean_text(self, text):
        """Membersihkan teks dari karakter tidak penting"""
        # Hapus URL
        text = re.sub(r'http\S+', '', text)
        # Hapus karakter khusus
        text = re.sub(r'[^\w\s]', '', text)
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        return text.strip()

    def translate_text(self, text):
        """Terjemahkan teks bahasa Indonesia ke Inggris (sederhana)"""
        # Ini adalah implementasi sederhana, sebaiknya gunakan library terjemahan sebenarnya
        # Untuk produksi, pertimbangkan menggunakan googletrans atau layanan API
        id_to_en = {
            'positif': 'positive',
            'negatif': 'negative',
            'naik': 'rise',
            'turun': 'fall',
            'bagus': 'good',
            'buruk': 'bad',
            'untung': 'profit',
            'rugi': 'loss',
            'beli': 'buy',
            'jual': 'sell'
        }
        
        for id_word, en_word in id_to_en.items():
            text = text.replace(id_word, en_word)
            
        return text

    def analyze_articles(self, articles):
        """Menganalisis sentimen semua artikel"""
        for article in articles:
            text = f"{article['title']}. {article['description']}"
            sentiment = self.analyze_sentiment(text)
            article['sentiment'] = sentiment
            
            # Kategorikan sentimen
            score = sentiment['combined_score']
            if score > 0.1:
                article['sentiment_label'] = 'Positif'
            elif score < -0.1:
                article['sentiment_label'] = 'Negatif'
            else:
                article['sentiment_label'] = 'Netral'
                
        return articles

    def summarize_sentiment(self, articles):
        """Ringkasan sentimen dari semua artikel"""
        if not articles:
            return {
                'total_articles': 0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'average_score': 0
            }
        
        total_score = 0
        sentiment_count = {'Positif': 0, 'Netral': 0, 'Negatif': 0}
        
        for article in articles:
            sentiment = article.get('sentiment_label', 'Netral')
            sentiment_count[sentiment] += 1
            total_score += article['sentiment']['combined_score']
            
        return {
            'total_articles': len(articles),
            'positive': sentiment_count['Positif'],
            'neutral': sentiment_count['Netral'],
            'negative': sentiment_count['Negatif'],
            'average_score': total_score / len(articles)
        }
