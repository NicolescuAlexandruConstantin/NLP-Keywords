from pipeline import pipeline
from algorithms.TextRank import TextRank
from algorithms.RAKE import RAKE
from algorithms.TF_IDF import TFIDF
from algorithms.YAKE import YAKE

class Main:
    def __init__(self):
        self.pipeline = pipeline(TextRank(),TFIDF(),YAKE(),RAKE())
    def run(self):
        text=""
        self.pipeline.process(text)


main=Main()
main.run()