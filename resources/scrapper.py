import time
import json
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
from newspaper import Article, Config
from langchain_community.tools import TavilySearchResults

        

class WebContentExtractor:
    """Extracts text content and metadata from web pages."""

    @staticmethod
    def get_text_content(
        url: str
    ) -> str:
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
    def get_title(
        url: str
    ) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.title.string.strip() if soup.title else "Title not found"
        except requests.RequestException as e:
            return f"[Error] Web request failed: {e}"

    @staticmethod
    def format_text(
        text: str
    ) -> str:
        return " ".join(text.split())


class CustomWebScraper:
    """Performs web searches and extracts relevant data."""

    def __init__(self, max_results: int = 10):
        self.tool = TavilySearchResults(
            max_results= max_results,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )
        self.results_dict: Dict[str, Dict[str, str]] = {}

    def process_results(self, queries: List[str]) -> List[Dict[str, str]]:
        """Processes search results, filters out errors, and extracts content."""
        
        for query in queries:
            print(f"---> Searching: {query}")
            result_raw = self.tool.invoke({"query": query})

            results_list = []
            if isinstance(result_raw, str):
                try:
                    results_list = json.loads(result_raw)
                except json.JSONDecodeError as e:
                    print(f"---> Error parsing JSON for query '{query}': {e}")
                    continue
            elif isinstance(result_raw, list):
                results_list = result_raw

            for res in results_list:
                url = res.get("url", "").strip()
                if not url or url in self.results_dict:
                    continue 

                snippet = res.get("content", "")
                full_title = WebContentExtractor.get_title(url)
                full_content_raw = WebContentExtractor.get_text_content(url)
                full_content = WebContentExtractor.format_text(full_content_raw) if full_content_raw else ""

                if "[Error]" in full_content:
                    print(f"---> [Error] Skipping invalid content from URL: ")
                    continue

                # Store valid results
                self.results_dict[url] = {
                    "query": query,
                    "direct_snippet": snippet,
                    "url": url,
                    "full_title": full_title,
                    "full_content": full_content,
                }

            time.sleep(1)

        final_results = list(self.results_dict.values())
        print(f"-----> Total valid results obtained: {len(final_results)}")
        return final_results