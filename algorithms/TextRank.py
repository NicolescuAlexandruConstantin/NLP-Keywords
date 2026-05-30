from typing import Dict, List, Set, Union
import numpy as np
import spacy
from algorithms.Commons import Commons

class TextRank(Commons):
    """Keyword extraction using the TextRank graph-based algorithm."""

    def extract(
            self,
            doc: spacy.tokens.Doc,
            top_k: int = 10,
            window: int = 4,
            d: float = 0.85,
            convergence_threshold: float = 1e-5,
    ) -> List[Dict[str, Union[float, str]]]:
        """Execute the TextRank pipeline to extract top-k keywords."""

        # Step 1: Extract candidate lemmas (nouns and adjectives)
        candidate_lemmas = []
        for token in doc:
            if token.pos_ in {"NOUN", "ADJ"} and token.is_alpha:
                norm_lemma = self._normalize(token)
                if norm_lemma not in self.stopwords:
                    candidate_lemmas.append(norm_lemma)

        if not candidate_lemmas:
            return []

        # Step 2: Build graph (co-occurrence within window)
        vocab_list = list(dict.fromkeys(candidate_lemmas))
        word_to_index = {word: i for i, word in enumerate(vocab_list)}
        adjacency_matrix = np.zeros((len(vocab_list), len(vocab_list)), dtype=np.float32)

        for i in range(len(candidate_lemmas)):
            for j in range(i + 1, min(i + window, len(candidate_lemmas))):
                a, b = word_to_index[candidate_lemmas[i]], word_to_index[candidate_lemmas[j]]
                if a == b:
                    continue
                adjacency_matrix[a, b] += 1
                adjacency_matrix[b, a] += 1

        # Step 3: Normalize matrix and run PageRank algorithm
        row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(adjacency_matrix, row_sums, where=row_sums != 0)
        scores = np.random.rand(len(vocab_list)).astype(np.float32)

        while True:
            prev_scores = scores.copy()
            scores = (1 - d) + d * transition_matrix.T @ prev_scores
            if np.linalg.norm(scores - prev_scores, 1) < convergence_threshold:
                break

        word_scores = dict(zip(vocab_list, scores))

        # Step 4: Combine adjacent words into phrases and score them
        phrase_scores = {}
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.pos_ in {"NOUN", "ADJ"} and token.is_alpha:
                norm1 = self._normalize(token)
                if norm1 in word_scores:
                    if i + 1 < len(doc):
                        token2 = doc[i + 1]
                        norm2 = self._normalize(token2)

                        if token2.pos_ in {"NOUN", "ADJ"} and token2.is_alpha and norm2 in word_scores:
                            phrase = f"{norm1} {norm2}"
                            score = word_scores[norm1] + word_scores[norm2]
                            phrase_scores[phrase] = score
                            i += 2
                            continue

                    phrase_scores[norm1] = word_scores[norm1]
            i += 1

        # Step 5: Sort phrases and return values
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        top_phrases = [
            {"keyword": phrase, "score": round(float(score), 4)}
            for phrase, score in sorted_phrases[:top_k]
        ]

        return top_phrases