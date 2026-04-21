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
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters (very common noise)
    text = text.replace("\u200c", "").replace("\u200d", "")

    # Strip extra whitespace
    text = " ".join(text.split())

    return text


# Image preprocessing

import cv2
import numpy as np

def preprocess_image(image):
    # Convert PIL to OpenCV compatible format 
    image = np.array(image)

    # Convert to grayscale
    # OCR only cares about structure not color, converting to gray scale only changes the color but doesn't affect the edges of characters, strokes, shapes, contrast between text and background. It reduces data from 3 channel to 2 channel making computation faster lighter and easier to process in bulk pipelines. OCR models rely heavily on luminance contrast like black text on white background or dark text on light background and most OCR preprocessing pipelines assumes grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Contrast Normalization
    image = cv2.equalizeHist(image)
    
    # Resize to OCR friendly fixed size
    h, w = image.shape
    new_h = 64
    new_w = int(w * (new_h / h))
    image = cv2.resize(image, (new_w, new_h))

    # Normalize the pixel value
    image = image.astype(np.float32)  # ensure float
    image = image / 255.0
    # image = (image - 0.5) / 0.5

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

