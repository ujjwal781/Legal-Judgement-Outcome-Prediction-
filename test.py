from collections import Counter
from datasets import load_dataset

labels = [item["label"] for item in dataset["train"]]
print(Counter(labels))