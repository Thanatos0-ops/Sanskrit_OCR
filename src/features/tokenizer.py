"""
Extract Unique characters from dataset
Map characters to integer (encoding)
Map integer to character (decoding)
Add CTC-specific token 
# CTC (Connectionist Temporal Classification) is a loss function used in OCR to train models without needing exact alignment between image parts and characters.
Example:
If a model predicts --HH-EE-LL-LL-OO-- (where - = blank), CTC collapses it to “HELLO” by removing repeats and blanks.
"""


# Step 1: Build Vocabulary

def build_vocab(dataset):
    """
    Builds a character level vocabulary from dataset.

    Args:
        dataset(list): list of samples {"image": ..., "text":...}

    Returns:
        set: Unique characters
    """
    vocab = set()

    for sample in dataset:
        text = sample["text"]
        for char in text:
            vocab.add(char)

    return vocab



# Step 2: Create Mapping (CTC compatible)
# CTC needs: A blank token, Stable Indexing

def create_mapping(vocab):
    """
    Creates char-to-index and index-to-char mappings.
    
    Adds CTC blank token at index 0.
    """

    # Sort for consistency 
    vocab = sorted(list(vocab))

    char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}
    char2idx["<BLANK>"] = 0

    idx2char = {idx: char for char, idx in char2idx.items()}

    return char2idx, idx2char



# Step 3: Encoding text to numbers

def encode_text(text, char2idx):
    """
    Converts text string to list of integer indices.
    """
    return [char2idx[char] for char in text if char in char2idx]



# Step 4: Decoding (for predicition later)
# CTC decoding requires: Removing repeated characters, Removing blanks

def decode_indices(indices, idx2char):
    """
    Decodes indices to text (CTC style)
    Removes blanks and repeated characters
    """

    decoded = []
    prev = None
    
    for idx in indices:
        if idx == 0: # Blank
            prev = None
            continue

        if idx != prev:
            decoded.append(idx2char[idx])
            prev = idx
        

    return "".join(decoded)


# Step 5: Wrap into a class

class Tokenizer:
    def __init__(self, dataset):
        vocab = build_vocab(dataset)
        self.char2idx, self.idx2char = create_mapping(vocab)

    def encode(self, text):
        return encode_text(text, self.char2idx)
    
    def decode(self, indices):
        return decode_indices(indices, self.idx2char)
    
    def vocab_size(self):
        return len(self.char2idx)
