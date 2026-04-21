from datasets import load_dataset
from preprocess import process_sample
import tqdm

def load_data():
    return load_dataset("snskrt/Sanskrit_OCR_Parallel_Corpus")

def build_dataset():
    dataset = load_data()

    processed = []

    for sample in tqdm(dataset["train"]):
        try:
            processed.append(process_sample(sample))
        except Exception:
            continue
    
    return processed


if __name__ == "__main__":
    data = build_dataset()
    print("Processed Samples: ", len(data))

