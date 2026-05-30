import json
import math
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Union
import spacy
from algorithms.Commons import Commons


class TFIDF(Commons):
    """Keyword extraction using Term Frequency - Inverse Document Frequency."""

    def __init__(self, nlp: spacy.language.Language, stopwords: Set[str]) -> None:
        super().__init__(nlp, stopwords)
        self.idf: Dict[str, float] = {}
        self.default_idf: float = 0.0

    def _get_lemmas(self, text: str) -> List[str]:
        """Helper to extract clean, non-stopword lemmas from text."""
        doc = self.nlp(text)
        lemmas = []
        for token in doc:
            if token.is_alpha:
                norm = self._normalize(token)
                if norm not in self.stopwords:
                    lemmas.append(norm)
        return lemmas


    def build_idf_from_directory(self, directory_path: str) -> None:
        """Reads all .txt files in a directory to compute and store IDF scores."""
        doc_count = 0
        df = Counter()

        path = Path(directory_path)
        txt_files = list(path.glob("*.txt"))

        if not txt_files:
            raise ValueError(f"No .txt files found in the directory: {directory_path}")

        print(f"Processing {len(txt_files)} documents to build IDF...")

        for filepath in txt_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            unique_lemmas = set(self._get_lemmas(text))
            for lemma in unique_lemmas:
                df[lemma] += 1
            doc_count += 1

        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log((1 + doc_count) / (1 + freq)) + 1


        self.default_idf = math.log((1 + doc_count) / 1) + 1

    def save_idf(self, filepath: str) -> None:
        """Saves the computed IDF dictionary to a JSON file."""
        if not self.idf:
            raise ValueError("No IDF data to save. Run build_idf_from_directory() first.")

        data = {
            "default_idf": self.default_idf,
            "idf": self.idf
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved IDF scores to {filepath}")

    def load_idf(self, filepath: str) -> None:
        """Loads a pre-computed IDF dictionary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.default_idf = data.get("default_idf", 1.0)
        self.idf = data.get("idf", {})


    def extract(self, text: str, top_k: int = 10) -> List[Dict[str, Union[float, str]]]:
        """Calculates TF for a new document and multiplies by pre-computed IDFs."""
        if not self.idf and not self.default_idf:
            raise ValueError("IDF vocabulary not loaded. Call load_idf() first.")

        lemmas = self._get_lemmas(text)
        if not lemmas:
            return []

        total_terms = len(lemmas)
        tf_counts = Counter(lemmas)

        tf_idf_scores = {}
        for word, count in tf_counts.items():
            tf = count / total_terms

            idf = self.idf.get(word, self.default_idf)

            tf_idf_scores[word] = tf * idf

        sorted_words = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {"keyword": word, "score": round(score, 4)}
            for word, score in sorted_words[:top_k]
        ]