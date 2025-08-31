# Use a pipeline as a high-level helper
from transformers import pipeline
import pickle

# pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")



# pipe = pipeline("text-classification", model="borisn70/bert-43-multilabel-emotion-detection")

# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("image-classification", model="prithivMLmods/Facial-Emotion-Detection-SigLIP2")

pipe = pipeline("text-classification", model="boltuix/NeuroFeel")

from PIL import Image

img = Image.open("Screenshot 2025-06-13 032239.png")
result = pipe(img)
print(result)

