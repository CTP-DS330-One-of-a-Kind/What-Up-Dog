from flask import Flask, render_template, request, jsonify
from fastai.vision.all import load_learner, PILImage
import pathlib
from openai import OpenAI
import tensorflow as tf
from transformers import pipeline
import numpy as np
import os
from werkzeug.utils import secure_filename
from util import load_models, model_operations

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} #heic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# load models
model = load_models.load_emotion_classifier() 
img_to_txt = load_models.load_image_to_text()
client = load_models.load_gpt3()

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Read the image file
        img = model_operations.preprocess_image(file)

        context = model_operations.get_context(img_to_txt, img)

        predicted_result = model_operations.predict_emotion(model, img)
        
        if not model_operations.dog_precheck(context):
            not_a_dog = "THAT'S NOT A DOG! Or our model didn't detect a dog, in which case we profusely apologize :("
            return jsonify({'prediction': not_a_dog, 'file_path': file_path})
    
        emotion = predicted_result
    
        response_content = model_operations.generate_dog_text(client, emotion, context)
        
        print(type(model))
        print(type(img_to_txt))
        print(type(client)) 
        print(response_content)
        return jsonify({'prediction': str(response_content), 'file_path': file_path})

if __name__ == '__main__':
    app.run(debug=True)