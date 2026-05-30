import os
import spacy
from algorithms.TF_IDF import TFIDF
from stopword_manager import StopWordManager

## ALGORITM CREAT STRICT PENTRU O PRIMA RULARE A TF-IDF, PENTRU A CREA VALORILE IDF
## ASTFEL INCAT SA NU TREBUIASCA SA SE PARCURGA TOATE DOCUMENTELE LA FIECARE RULARE


print("Loading spaCy model...")
nlp = spacy.load("ro_core_news_lg")

stopwords = StopWordManager.load()

tfidf_extractor = TFIDF(nlp, stopwords)

idf_cache_path = "ro_idf_cache.json"
dataset_directory = "./dataset"
if os.path.exists(idf_cache_path):
    print("IDF Cache Exists")
else:
    print("Building IDF scores from corpus for the first time...")
    tfidf_extractor.build_idf_from_directory(dataset_directory)
    tfidf_extractor.save_idf(idf_cache_path)

new_text = "Acesta este un document nou despre inteligența artificială și procesarea limbajului."
keywords = tfidf_extractor.extract(new_text, top_k=5)

print(keywords)