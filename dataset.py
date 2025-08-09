import json
from datasets import Dataset

def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

if __name__ == "__main__":
    ds = load_dataset("../data/train.json")
    print(ds[0])
