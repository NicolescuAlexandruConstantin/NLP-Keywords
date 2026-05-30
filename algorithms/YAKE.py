from algorithms.Commons import Commons
from collections import Counter
from typing import Dict, List, Union
from utils import Normalizer


class YAKE(Commons):

    def _extract_candidates(self, text: str) -> List[List[str]]:
        """
        Extract candidate phrases (1-3 words).
        """
        doc = self.nlp(text)

        candidates = []
        current_phrase = []

        for token in doc:
            word = self._normalize(token)

            if token.is_alpha and word not in self.stopwords:
                current_phrase.append(word)
            else:
                if current_phrase:
                    candidates.append(current_phrase)
                    current_phrase = []

        if current_phrase:
            candidates.append(current_phrase)

        final_candidates = []

        for phrase in candidates:
            for n in range(1, min(4, len(phrase) + 1)):
                for i in range(len(phrase) - n + 1):
                    final_candidates.append(phrase[i:i+n])

        return final_candidates

    def _compute_word_scores(
        self,
        candidates: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute YAKE-inspired scores for words.
        Smaller score = more important.
        """

        word_freq = Counter()
        first_position = {}

        total_words = 0

        for phrase in candidates:
            for word in phrase:

                if word not in first_position:
                    first_position[word] = total_words + 1

                word_freq[word] += 1
                total_words += 1

        word_scores = {}

        for word, freq in word_freq.items():

            position_score = first_position[word] / max(total_words, 1)

            length_score = 1 / (len(word) + 1)

            frequency_score = 1 / freq

            score = (
                position_score
                * frequency_score
                * (1 + length_score)
            )

            word_scores[word] = score

        return word_scores

    def _compute_candidate_scores(
        self,
        candidates: List[List[str]],
        word_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        YAKE combines word scores.
        Lower score = better.
        """

        scores = {}

        for phrase in candidates:

            phrase_text = " ".join(phrase)

            score = 1.0

            for word in phrase:
                score *= word_scores[word]

            scores[phrase_text] = score

        return scores

    def extract(
        self,
        text: str,
        top_k: int = 10
    ) -> List[Dict[str, Union[float, str]]]:

        candidates = self._extract_candidates(text)

        if not candidates:
            return []

        word_scores = self._compute_word_scores(candidates)

        candidate_scores = self._compute_candidate_scores(
            candidates,
            word_scores
        )

        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1]
        )

        seen_norms = set()
        keywords = []

        for phrase, score in sorted_candidates:

            normalized = Normalizer.norm(phrase)

            if normalized in seen_norms:
                continue

            seen_norms.add(normalized)

            keywords.append({
                "keyword": phrase,
                "score": round(float(score), 6)
            })

            if len(keywords) >= top_k:
                break

        return keywords