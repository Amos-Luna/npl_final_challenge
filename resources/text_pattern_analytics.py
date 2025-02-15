import spacy
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TextPatternAnalyzer:
    """
    Extracts named entities, POS tagging, and sentiment analysis.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        text_column: str, 
        language_model: str = "es_core_news_sm"
    )-> None:
        """Initializes with a DataFrame, text column, and a language model."""
        self.df = df
        self.text_column = text_column
        self.nlp = spacy.load(language_model)  # Load the Spanish model
        self.vader_analyzer = SentimentIntensityAnalyzer() 
        
    def extract_named_entities(self):
        """Extracts named entities using spaCy."""
        self.df['named_entities'] = self.df[self.text_column].apply(
            lambda text: [(ent.text, ent.label_) for ent in self.nlp(text).ents]
        )
    
    def extract_pos_tags(self):
        """Extracts POS tags using spaCy."""
        self.df['pos_tags'] = self.df[self.text_column].apply(
            lambda text: [(token.text, token.pos_) for token in self.nlp(text)]
        )
    
    def analyze_sentiment_textblob(self):
        """Performs sentiment analysis using TextBlob."""
        self.df['sentiment_TextBlob'] = self.df[self.text_column].apply(
            lambda text: TextBlob(text).sentiment.polarity
        )
        self.df['sentiment_label_TextBlob'] = self.df['sentiment_TextBlob'].apply(
            lambda score: "positive" if score > 0 else ("negative" if score < 0 else "neutral")
        )

    def analyze_sentiment_vader(self):
        """Performs sentiment analysis using VADER."""
        self.df['sentiment_VADER'] = self.df[self.text_column].apply(
            lambda text: self.vader_analyzer.polarity_scores(text)['compound']
        )
        self.df['sentiment_label_VADER'] = self.df['sentiment_VADER'].apply(
            lambda score: "positive" if score > 0.2 else ("negative" if score < -0.2 else "neutral")
        )
    
    def run_full_analysis(self):
        """Runs all analysis steps: NER, POS, and both sentiment analyses."""
        print("\nExtracting Named Entities...")
        self.extract_named_entities()
        
        print("\nExtracting POS Tags...")
        self.extract_pos_tags()
        
        print("\nPerforming Sentiment Analysis with TextBlob...")
        self.analyze_sentiment_textblob()
        
        print("\nPerforming Sentiment Analysis with VADER...")
        self.analyze_sentiment_vader()
        
        print("\nAnalysis Completed!")
        return self.df