from typing import Dict, List, Set, Union
import spacy
from utils import Normalizer

class Commons:
    def __init__(self, nlp: spacy.language.Language, stopwords: Set[str]) -> None:
        self.nlp = nlp
        self.stopwords = stopwords

    def _normalize(self, token: spacy.tokens.Token) -> str:
        """Normalize a token using lemmatization."""
        return Normalizer.norm(token.lemma_)