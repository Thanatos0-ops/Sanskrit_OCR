"""
A PyTorch-style pipeline:
1. Dataset class -> returns(Image, encoded_text)
2. Collate function -> handles batching, padding
3. DataLoader -> feeds model
"""


# Step 1: Dataset class
# Connects: your processed data, your tokenizer

import torch
from torch.utils.data import Dataset

class ORCDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = sample['image']
        text = sample['text']
    
        # Convert to tensor
        image = torch.tensor(image, dtype = torch.float32)

        # Add channel dimension -> (1, H, W)
        image = image.unsqueeze(0)

        # Encode text
        encoded_text = self.tokenizer.encode(text)

        return image, torch.tensor(encoded_text, dtype=torch.long)



# Step 2: Collagte function
# Problem: images have different widths, Text sequences have different lenght, CTC needs lengths

def collate_fn(batch):
    """
    batch = list of (image, text_tensor)
    """

    images, texts = zip(*batch)

    # Handle images (pad width)
    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]

    max_w = max(widths)

    padded_images = []
    for img in images:
        c, h, w = img.shape

        pad_w = max_w - w

        # Pad on right side
        padded = torch.nn.functional.pad(img, (0, pad_w), value=0)
        padded_images.append(padded)

    images = torch.stack(padded_images)

    # Handle text (Flatten for CTC)
    text_lenghts = torch.tensor([len(t) for t in texts], dtype = torch.long)

    # Concatenate all text into 1D
    texts = torch.cat(texts)

    return images, texts, text_lenghts

# Step 3: DataLoader wrapper

from torch.utils.data import DataLoader

def get_dataloader(data, tokenizer, batch_size=8, shuffle=True):
    dataset = ORCDataset(data, tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )