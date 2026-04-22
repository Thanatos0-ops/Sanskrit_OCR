"""
Pipeline:
    image(B, 1, H, W)
        CNN (Feature Extractor)
        reshape (B, C, H, W -> B, W, features)
        BiLSTM (sequence modeling)
        Linear layer
        CTC logits


Key idea: Width = time dimension

"""

# Step 1: CNN feature Extractor
# Reduce height, Preserve width as much as possible

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
             nn.Conv2d(1, 64, 3, padding=1),   # (B,64,H,W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 64, H/2, W/2)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 128, H/4, W/4)

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),              # reduce height only

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2,1)),              # reduce height only
        )

    def forward (self, x):
            return self.conv(x)
        

# Step 2: Full OCR Model

class OCRModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.cnn = CNN()

        # BiLSTM input size depends on the CNN output
        self.rnn = nn.LSTM(
            input_size=512,     # channels after CNN
            hidden_size=256,
            num_layers=2,
            bidirectional=True, 
            batch_first=True
        )

        self.fc = nn.Linear(512, vocab_size) # 256*2 (bidirectional)

    
    def forward(self, x):
        # x: (B, 1, H, W)

        x = self.cnn(x)
        # (B, C=512, H', W')

        b, c, h, w = x.size()

        # Collapse height dimension
        # Convert the CNN output into sequence format
        x = x.permute(0, 3, 1, 2)   # (B, W, C, H)
        x = x.view(b, w, c * h)     # (B, W, features)

        # Now sequence = width
        x, _ = self.rnn(x)      # (B, W, 512)

        x = self.fc(x)      # (B, W, vocab_size)

        # CTC expects (W, B, vocab)
        x = x.permute(1, 0, 2)

        return x