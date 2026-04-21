import pickle
import os

def save_dataset(data, path= "data/processed/orc.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok = True)

    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_dataset(path="data/processed/orc.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)