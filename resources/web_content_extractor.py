import requests
import re
import time
from bs4 import BeautifulSoup
from newspaper import Article, Config

class WebContentExtractor:
    """Extrae texto y metadatos de páginas web."""
    
    @staticmethod
    def get_text_content(url: str) -> str:
        """Extrae el contenido completo de un artículo desde una URL."""
        try:
            config = Config()
            config.browser_user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/85.0.4183.102 Safari/537.36"
            )
            article = Article(url, config=config)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            return f"[Error] Extracting content: {e}"

    @staticmethod
    def get_title(url: str) -> str:
        """Obtiene el título de la página web."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.title.string.strip() if soup.title else "Title not found"
        except requests.RequestException as e:
            return f"[Error] Web request failed: {e}"

    @staticmethod
    def clean_text(text: str) -> str:
        """Limpia el contenido de texto eliminando espacios extras y HTML."""
        return re.sub(r'\s+', ' ', BeautifulSoup(text, "html.parser").get_text()).strip()
