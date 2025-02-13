import pandas as pd
import spacy
import torch
from collections import Counter
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedTextAnalyzer:
    """
    Performs advanced NLP analysis using Deep Learning and Topic Modeling.
    """
    
    def __init__(self, df: pd.DataFrame, text_column: str):
        self.df = df
        self.text_column = text_column
        self.ner_model = spacy.load("es_core_news_sm")  # SpaCy model for Spanish
        self.sentiment_model = pipeline("text-classification", 
                                        model="nlptown/bert-base-multilingual-uncased-sentiment", 
                                        tokenizer="nlptown/bert-base-multilingual-uncased-sentiment")
        
        
    def perform_sentiment_analysis(self):
        """Applies BERT-based sentiment analysis on the dataset."""
        print("Performing Sentiment Analysis using BERT...")
        self.df['sentiment'] = self.df[self.text_column].apply(lambda x: self.sentiment_model(x, truncation=True)[0]['label'])
        sentiment_counts = self.df['sentiment'].value_counts()
        print("\nSentiment Distribution:")
        print(sentiment_counts)
        
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm')
        plt.title("Sentiment Analysis Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.show()
    
    def perform_ner_analysis(self):
        """Extracts Named Entities and visualizes the most frequent ones."""
        print("Extracting Named Entities...")
        all_entities = []
        
        for doc in self.df[self.text_column].dropna():
            spacy_doc = self.ner_model(doc)
            all_entities.extend([ent.text for ent in spacy_doc.ents])
        
        entity_counts = Counter(all_entities).most_common(10)
        print("\nTop Named Entities:")
        for entity, count in entity_counts:
            print(f"{entity}: {count}")
        
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=[ent[0] for ent in entity_counts], y=[ent[1] for ent in entity_counts], palette='viridis')
        plt.title("Most Frequent Named Entities")
        plt.xticks(rotation=45)
        plt.xlabel("Entities")
        plt.ylabel("Count")
        plt.show()
    
    def perform_topic_modeling(
        self, 
        num_topics=5
    ):
        """Applies LDA Topic Modeling to discover hidden topics in the dataset."""
        print(f"Performing Topic Modeling with {num_topics} topics...")
        #vectorizer = CountVectorizer(stop_words='spanish')
        vectorizer = CountVectorizer(stop_words=None)
        doc_term_matrix = vectorizer.fit_transform(self.df[self.text_column].dropna())
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(doc_term_matrix)
        
        words = vectorizer.get_feature_names_out()
        print("\nTop words per topic:")
        for topic_idx, topic in enumerate(lda_model.components_):
            print(f"\nTopic {topic_idx+1}:")
            print(" ".join([words[i] for i in topic.argsort()[-10:]]))
        
        
    def run_full_advanced_analysis(self):
        """Runs all advanced NLP analyses."""
        self.perform_sentiment_analysis()
        #self.perform_ner_analysis()
        #self.perform_topic_modeling(num_topics=5)
