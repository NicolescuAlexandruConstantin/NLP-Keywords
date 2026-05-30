from typing import Any, Dict, Set
import spacy

from algorithms.TF_IDF import TFIDF
from stopword_manager import StopWordManager
from algorithms.TextRank import TextRank
from algorithms.RAKE import RAKE
from utils import Normalizer

class pipeline:
    def __init__(self) -> None:
        self.nlp = spacy.load("ro_core_news_lg")

        # Load stopwords
        self.stopwords: Set[str] = StopWordManager.load()

        # Keyword extractor (RAKE + TextRank)
        self.textrank = TextRank(self.nlp, self.stopwords)
        self.rake = RAKE(self.nlp, self.stopwords)
        self.tfidf = TFIDF(self.nlp, self.stopwords)

    # Normalize and split all keywords into individual tokens
    def _token_set(self, keywords: Set[str]) -> Set[str]:
        return {Normalizer.norm(w) for kw in keywords for w in kw.split()}

    # Main method to perform text analysis and return all results
    def analyze(self, txt: str) -> Dict[str, Any]:
        doc = self.nlp(txt)

        # Keyword Extraction
        rake_keywords = self.rake.extract(txt)
        textrank_keywords = self.textrank.extract(doc)

        # Final Result
        return {
            "extractedText": txt,
            "rake": rake_keywords,
            "textrank": textrank_keywords
        }
