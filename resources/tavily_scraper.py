import os
import json
import time
from typing import List, Dict
from tavily import TavilyClient
from resources.web_content_extractor import WebContentExtractor

class TavilyScraper:
    """Realiza búsquedas en la web usando Tavily y extrae datos relevantes."""

    def __init__(self, api_key: str = None, max_results: int = 10):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY no encontrado.")
        
        self.client = TavilyClient(self.api_key)
        self.max_results = max_results

    def search_tavily(self, queries: List[str]) -> List[Dict[str, str]]:
        """Realiza búsquedas en Tavily y filtra PDFs y duplicados."""
        formatted_results = []
        urls_seen = set()

        for query in queries:
            print(f"🔎 Buscando en Tavily: {query}")
            response = self.client.search(query=query, search_depth="advanced", max_results=self.max_results)

            for res in response.get("results", []):
                url = res.get("url", "").strip()
                if not url or url in urls_seen or url.lower().endswith(".pdf"):
                    continue  # Evitamos duplicados y PDFs

                urls_seen.add(url)
                formatted_results.append({
                    "query": query,
                    "direct_snippet": res.get("content", "No snippet available"),
                    "url": url,
                    "full_title": WebContentExtractor.get_title(url),
                    "full_content": WebContentExtractor.get_text_content(url),
                })

            time.sleep(1)  # Para evitar bloqueos de la API

        print(f"Total de resultados obtenidos: {len(formatted_results)}")
        return formatted_results

    def save_results(self, results: List[Dict[str, str]], filename: str):
        """Guarda los resultados en un archivo JSON."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Datos guardados en {filename}")

if __name__ == "__main__":
    print("TavilyScraper.")
