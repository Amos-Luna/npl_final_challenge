import json
import pandas as pd
import os
from datetime import datetime
from resources.text_processor import TextCleaner, TextProcessor


def save_data(
    results: list,
    file_name: str
) -> None:
    """Save extracted data in a JSON file with proper encoding and timestamp."""

    date_stamp = datetime.now().strftime("%d%m%Y")
    path = f"../data/data_extracted_{file_name}_{date_stamp}_new.json"
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_json:
        json.dump(results, f_json, ensure_ascii=False, indent=2)

    print(f"-----> Data successfully saved in: {path}")
    
def combine_text(row):
    """Combina las columnas 'direct_snippet' y 'full_content'."""
    direct = row.get('direct_snippet', '')
    full = row.get('full_content', '')
    return f"{direct} {full}".strip()


def preprocess_dataframe(
    df: pd.DataFrame, 
    text_column: str
) -> pd.DataFrame:
    """
    Applies text cleaning and processing to a DataFrame column, creating multiple intermediate columns.
    """
    cleaner = TextCleaner()
    processor = TextProcessor()

    if 'direct_snippet' in df.columns and 'full_content' in df.columns:
        df['combined_text'] = df.apply(combine_text, axis=1)
        text_column = 'combined_text'

    df['cleaned_text'] = df[text_column].apply(cleaner.clean_text)
    df['tokens'] = df['cleaned_text'].apply(processor.tokenize_text)
    df['no_stopwords'] = df['tokens'].apply(processor.remove_stopwords)
    df['stemmed'] = df['no_stopwords'].apply(processor.apply_stemming)
    df['lemmatized'] = df['stemmed'].apply(lambda tokens: processor.apply_lemmatization(' '.join(tokens)))
    df['final_text'] = df['lemmatized'].apply(lambda tokens: ' '.join(tokens))
    
    return df