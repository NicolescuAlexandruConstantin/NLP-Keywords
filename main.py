import os
import pipeline


class Main:
    def __init__(self):
        self.output_dir = "results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def format_output(self, results: dict) -> None:
        """Takes the raw pipeline dictionary and prints a clean terminal report."""
        print("\n" + "=" * 55)
        print("NLP PIPELINE RESULTS")
        print("=" * 55)

        for algo_name, data in results.items():
            if algo_name == 'extractedText':
                continue

            print(f"\n {algo_name.upper()} KEYWORDS")
            print("-" * 55)

            if not data:
                print("No keywords extracted.")
                continue

            max_kw_len = max(len(item['keyword']) for item in data) + 4

            for rank, item in enumerate(data, start=1):
                keyword = item['keyword']
                score = item['score']
                print(f" {rank:>2}. {keyword:<{max_kw_len}} | Score: {score:.4f}")

        print("\n" + "=" * 55 + "\n")

    def save_to_files(self, results: dict) -> None:
        """Saves the complete keyword lists to individual text files."""
        print(f"Saving reports to the '{self.output_dir}' directory...")

        for algo_name, data in results.items():
            if algo_name == 'extractedText':
                continue

            safe_name = algo_name.replace("-", "_").lower()
            filepath = os.path.join(self.output_dir, f"{safe_name}_keywords.txt")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"=== {algo_name.upper()} KEYWORDS ===\n\n")

                if not data:
                    f.write("No keywords extracted.\n")
                    continue

                max_kw_len = max(len(item['keyword']) for item in data) + 4

                for rank, item in enumerate(data, start=1):
                    keyword = item['keyword']
                    score = item['score']
                    f.write(f"{rank:>3}. {keyword:<{max_kw_len}} | Score: {score:.4f}\n")

        print("All algorithm files saved successfully!\n")

    def start(self):
        a = pipeline.pipeline()

        with open("dataset/0.2.txt", "r", encoding="utf-8") as f:
            continut = f.read()

        print("Processing document...")
        w = a.analyze(continut, top_k=10)

        self.format_output(w)

        self.save_to_files(w)


x = Main()
x.start()