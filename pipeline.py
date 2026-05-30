from typing import Any, Dict, Set
import spacy
from stopword_manager import StopWordManager
from keyword_extractor import KeywordExtractor
from utils import Normalizer

class pipeline:
    def __init__(self) -> None:
        self.nlp = spacy.load("ro_core_news_lg")

        # Load stopwords
        self.stopwords: Set[str] = StopWordManager.load()

        # Keyword extractor (RAKE + TextRank)
        self.keyword_extractor = KeywordExtractor(self.nlp, self.stopwords)

    # Normalize and split all keywords into individual tokens
    def _token_set(self, keywords: Set[str]) -> Set[str]:
        return {Normalizer.norm(w) for kw in keywords for w in kw.split()}

    # Main method to perform text analysis and return all results
    def analyze(self, txt: str) -> Dict[str, Any]:
        doc = self.nlp(txt)

        # Keyword Extraction
        rake_keywords = self.keyword_extractor.rake(txt)
        textrank_keywords = self.keyword_extractor.textrank(doc)

        # Final Result
        return {
            "extractedText": txt,
            "rake": rake_keywords,
            "textrank": textrank_keywords
        }
