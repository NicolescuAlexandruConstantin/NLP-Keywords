import pipeline


class Main:
    def __init__(self):
        pass
    def start(self):
        a = pipeline.pipeline()
        with open("dataset/0.2.txt", "r", encoding="utf-8") as f:
            continut = f.read()
        print(continut)
        w = a.analyze(continut)
        print(w)

x = Main()
x.start()
