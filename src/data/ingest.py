from datasets import load_dataset
from src.data.preprocess import process_sample, validate_sample
from src.data.save import save_dataset
import tqdm

def load_data():
    return load_dataset("snskrt/Sanskrit_OCR_Parallel_Corpus")

def build_dataset():
    dataset = load_data()

    processed = []

    for sample in tqdm(dataset["train"]):
        if not validate_sample(sample):
            continue
        try:
            processed.append(process_sample(sample))
        except Exception:
            continue
    
    return processed


if __name__ == "__main__":
    data = build_dataset()
    print("Processed Samples: ", len(data))
    save_dataset(data)

