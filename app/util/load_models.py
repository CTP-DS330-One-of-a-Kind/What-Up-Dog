import pathlib
import platform
import os
from fastai.vision.all import load_learner
from transformers import pipeline
from openai import OpenAI

def load_emotion_classifier():
    model_path = pathlib.Path('model/model.pkl')
    if platform.system() == 'Windows':
        # Use WindowsPath on Windows machines
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            model = load_learner(model_path)
        finally:
            pathlib.PosixPath = posix_backup
    else:
        # Use default PosixPath on other operating systems
        model = load_learner(model_path)
    return model

def load_image_to_text():
    img_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return img_to_text

def load_gpt3():
    api = open("gpt_api.txt")
    os.environ["OPENAI_API_KEY"]= api.read()
    client = OpenAI()
    return client
