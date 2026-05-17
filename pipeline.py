class pipeline:
    def __init__(self, textrank, tfidf, yake, rake):
        self.textrank = textrank
        self.tfidf = tfidf
        self.yake = yake
        self.rake = rake

    def process(self, text):
        self.textrank.process(text)
        self.tfidf.process(text)
        self.yake.process(text)
        self.rake.process(text)
