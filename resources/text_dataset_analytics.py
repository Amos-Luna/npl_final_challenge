import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


class TextDatasetAnalyzer:
    """
    Performs exploratory analysis on a DataFrame containing processed text.
    """
    
    def __init__(
        self,
        df: pd.DataFrame, 
        text_column: str
    ) -> None:
        """Initializes with the DataFrame and the text column to analyze."""
        self.df = df
        self.text_column = text_column
        
    def get_basic_info(self):
        """Displays general information about the DataFrame."""
        
        print("\n--- General Information ---")
        print(f"Total number of examples: {len(self.df)}")
        print("\nMain characteristics:")
        print(self.df.info())
    
    
    def get_word_count_stats(self):
        """Calculates word count statistics."""
        self.df['word_count'] = self.df[self.text_column].apply(lambda x: len(str(x).split()))
        avg_words = self.df['word_count'].mean()
        print(f"\nAverage number of words per example: {avg_words:.2f}")
    
    
    def get_vocabulary_size(self):
        """Gets the vocabulary size."""
        all_words = ' '.join(self.df[self.text_column].dropna()).split()
        vocab_size = len(set(all_words))
        print(f"\nVocabulary size: {vocab_size}")
    
    
    def get_top_n_words(self, n=10):
        """Lists the most frequent words in the corpus."""
        all_words = ' '.join(self.df[self.text_column].dropna()).split()
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(n)
        print("\nMost frequent words:")
        for word, freq in top_words:
            print(f"{word}: {freq}")
    

    def compute_tfidf(self, top_n=10):
        """Computes TF-IDF scores and displays the top N most important words."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.df[self.text_column].dropna())
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1
        top_tfidf_words = sorted(zip(feature_names, avg_tfidf_scores), key=lambda x: x[1], reverse=True)[:top_n]
        
        print("\nTop TF-IDF Words:")
        for word, score in top_tfidf_words:
            print(f"{word}: {score:.4f}")
    
    def run_full_analysis(self):
        """Runs the full exploratory analysis including TF-IDF."""
        self.get_basic_info()
        self.get_word_count_stats()
        self.get_vocabulary_size()
        self.get_top_n_words(n=10)
        self.compute_tfidf(top_n=10)