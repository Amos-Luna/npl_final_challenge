import re
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


class TextCleaner:
    """
    Handles text cleaning operations such as removing punctuations, links, hashtags, numbers, etc.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def remove_punctuation(
        self, 
        text: str
    ) -> str:
        return re.sub(r'[^\w\s]', '', text)
    
    def remove_numbers(
        self, 
        text: str
    ) -> str:
        return re.sub(r'\d+', '', text)
    
    def remove_links(
        self, 
        text: str
    ) -> str:
        return re.sub(r'http\S+', '', text)
    
    def remove_hashtags(
        self, 
        text: str
    ) -> str:
        return re.sub(r'#\S+', '', text)
    
    def remove_users(
        self, 
        text: str
    ) -> str:
        return re.sub(r'@\S+', '', text)
    
    def remove_extra_spaces(
        self, 
        text: str
    ) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    def clean_text(
        self, 
        text: str
    ) -> str:
        """Applies all text cleaning functions."""
        
        text = text.lower()
        text = self.remove_links(text)
        text = self.remove_hashtags(text)
        text = self.remove_users(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuation(text)
        text = self.remove_extra_spaces(text)
        return text



class TextProcessor:
    """
    Handles text processing for Spanish, including tokenization, stopword removal, stemming, and lemmatization.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words("spanish"))
        self.stemmer = SnowballStemmer("spanish")
        self.nlp = spacy.load("es_core_news_sm") 
    
    def tokenize_text(
        self, 
        text: str
    ) -> List[str]:
        """Tokenizes the text into words."""
        
        return word_tokenize(text, language="spanish")
    
    def remove_stopwords(
        self, 
        tokens: List[str]
    ) -> List[str]:
        """Removes Spanish stopwords from tokenized text."""
        
        return [word for word in tokens if word.lower() not in self.stop_words]
    
    def apply_stemming(
        self, 
        tokens: List[str]
    ) -> List[str]:
        """Applies stemming to the tokens using SnowballStemmer for Spanish."""
        
        return [self.stemmer.stem(word) for word in tokens]
    
    def apply_lemmatization(
        self, 
        text: str
    ) -> List[str]:
        """Applies lemmatization using SpaCy for Spanish."""
        
        if isinstance(text, list):
            text = " ".join(text)
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]

    def process_text(
        self, 
        text: str
    ) -> str:
        """Applies full processing pipeline."""
        
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        stemmed_tokens = self.apply_stemming(tokens)
        lemmatized_tokens = self.apply_lemmatization(" ".join(stemmed_tokens))
        return " ".join(lemmatized_tokens)



