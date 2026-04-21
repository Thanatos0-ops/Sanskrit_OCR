# Step-1 :Normalization
import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize the input text using Unicode normalization (NFC).
    
    Args:
        text (str): The input text to be normalized.
        
    Returns:
        str: The normalized text.
    """

    # Ensures consistent representation of characters, especially for languages with complex scripts like Sanskrit. 
    return unicodedata.normalize('NFC', text)


# Image preprocessing

import cv2
import numpy as np

def preprocess_image(image):
    # Convert PIL to OpenCV compatible format 
    image = np.array(image)

    # Convert to grayscale
    # OCR only cares about structure not color, converting to gray scale only changes the color but doesn't affect the edges of characters, strokes, shapes, contrast between text and background. It reduces data from 3 channel to 2 channel making computation faster lighter and easier to process in bulk pipelines. OCR models rely heavily on luminance contrast like black text on white background or dark text on light background and most OCR preprocessing pipelines assumes grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to OCR friendly fixed size
    image = cv2.resize(image, (384, 64))

    # Normalize the pixel value

    img = img.astype(np.float32)  # ensure float
    img = img / 255.0
    # img = (img - 0.5) / 0.5

    return image

def validate_sample(sample):
    return(
        sample.get("image") is not None and
        sample.get("text") is not None and
        len(sample["text"].strip()) > 0
    )

def process_sample(sample):
    image = preprocess_image(sample["image"])
    text = normalize_text(sample["text"])

    return {
        "image" : image,
        "text": text
    }

