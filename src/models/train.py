"""
A proper training loop that handles:
    1. Forward Pass
    2. CTC Loss
    3. input/output lengths
    4. backprops
"""


# Step 1: Imports and setups

import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.data.save import load_dataset
from src.data.dataloader import get_dataloader
from src.features.tokenizer import Tokenizer
from src.models.model import OCRModel


# Step 2: Load data + tokenizer + dataloader

data = load_dataset()
tokenizer = Tokenizer(data)
dataloader = get_dataloader(data, tokenizer, batch_size=8)


# Step 3: Initialize model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OCRModel(tokenizer.vocab_size()).to(device)


# Step 4: CTC Loss 

criterion = nn.CTCLoss(blank=0, zero_infinity=True) # blank=0 matches the tokenizer, zero_infinity=True avoids NaN loss explosions


# Step 5: Optimizer

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Step 6: Training loop

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, texts, text_lengths in dataloader:

        images = images.to(device)
        texts = texts.to(device)
        text_lengths = text_lengths.to(device)

        # Forward Pass
        outputs = model(images)
        # shape: (T, B, vocab_size)

        # Critical Part: input_lengths (all same due to padding)
        batch_size = images.size(0)
        time_steps = outputs.size(0)

        input_lengths = torch.full(
            size = (batch_size,),
            fill_value = time_steps,
            dtype = torch.long
        ).to(device)

        # Compute loss
        loss = criterion(outputs, texts, input_lengths, text_lengths)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
    
    torch.save({
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": total_loss,
    }, f"models/checkpoints/epoch_{epoch+1}.pth")
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")