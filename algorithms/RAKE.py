from algorithms.Commons import Commons
from collections import Counter, defaultdict
from typing import Dict, List, Union
from utils import Normalizer

class RAKE(Commons):

    def _extract_phrases(self, text: str) -> List[List[str]]:
        """Extract candidate phrases (1 to 3 words, non-stopwords, alphabetic)."""
        doc = self.nlp(text)
        phrase_list = []
        current_phrase = []

        for token in doc:
            if token.is_alpha and self._normalize(token) not in self.stopwords:
                current_phrase.append(self._normalize(token))
            elif current_phrase:
                phrase_list.append(current_phrase)
                current_phrase = []

        if current_phrase:
            phrase_list.append(current_phrase)

        return [phrase for phrase in phrase_list if 1 <= len(phrase) <= 3]

    def _compute_word_scores(self, phrases: List[List[str]]) -> Dict[str, float]:
        """Compute word scores using frequency and degree."""
        word_freq = Counter()
        word_degree = defaultdict(int)

        for phrase in phrases:
            length = len(phrase)
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += length - 1

        return {word: (word_degree[word] + word_freq[word]) / word_freq[word] for word in word_freq}

    def _compute_phrase_scores(self, phrases: List[List[str]], word_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute phrase scores by summing up constituent word scores."""
        return {
            " ".join(phrase): sum(word_scores[word] for word in phrase)
            for phrase in phrases
        }

    def extract(self, text: str, top_k: int = 10) -> List[Dict[str, Union[float, str]]]:
        """Execute the RAKE pipeline to extract top-k keywords."""
        phrases = self._extract_phrases(text)
        if not phrases:
            return []

        word_scores = self._compute_word_scores(phrases)
        phrase_scores = self._compute_phrase_scores(phrases, word_scores)
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)

        seen_norms = set()
        final_keywords = []

        for phrase, score in sorted_phrases:
            normalized = Normalizer.norm(phrase)
            if normalized in seen_norms:
                continue
            seen_norms.add(normalized)
            final_keywords.append({"keyword": phrase, "score": round(float(score), 4)})
            if len(final_keywords) == top_k:
                break

        return final_keywords