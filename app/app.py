from flask import Flask, render_template, request, jsonify
from fastai.vision.all import load_learner, PILImage
import pathlib
from openai import OpenAI
import tensorflow as tf
from transformers import pipeline
import numpy as np
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#print("TEST ==========================================================")

#model_path = os.path.join(os.getcwd(),'model','model.pkl')
#print(model_path)
model_path = pathlib.Path('./model/model.pkl')
posix_backup = pathlib.PosixPath
try:
#     pathlib.PosixPath = pathlib.WindowsPath
     model = load_learner(model_path)
finally:
     pathlib.PosixPath = posix_backup

api = open("gpt_api.txt")
os.environ["OPENAI_API_KEY"]= api.read()
client = OpenAI()

def vit_to_string(result):
  string_result = ''.join(map(str,result))
  string_result = string_result[19:]
  return string_result

#precheck function see if dog is real or not
def precheck(streeng):
  if streeng.find('dog') != -1 or streeng.find('dogs') != -1 or streeng.find('puppy') != -1 or streeng.find('puppies') != -1:
    return True
  else:
    return False

#model = load_learner(model_path)


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
        img = PILImage.create(file)
        img.resize((288,288))

        # Make a prediction
        prediction = model.predict(img)

        # Extracting the predicted class
        predicted_result = prediction[0]

        img_array = np.array(img)

        # If using TensorFlow, you might need to expand dimensions 
        # and preprocess the input depending on your model requirements
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_to_txt = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        result = img_to_txt(img)

        emotion = predicted_result
        final_string = "You are a dog. If I were to take a picture of you right now you would be {}. Your tone and emotion would be considered {}".format(vit_to_string(result),emotion)

        gpt_dog = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": final_string},
            {"role": "user", "content": "Why do you think the dog in the picture is experiencing the emotion we have labeled it with and sent to you?"},
            {"role": "user", "content": "In 1 sentence as if you were a dog: explain why you would be feeling those emotions."}] )
    
        # I beat my head against a wall to parse out the message content, but I used gpt4 to get me to this
        # Check if 'choices' exists in the response and it has at least one element
        if hasattr(gpt_dog, 'choices') and len(gpt_dog.choices) > 0:
            # Extract the 'message' dictionary from the first element of 'choices'
            # This step depends on whether 'choices' is a list of dictionaries or a list of objects
            first_choice = gpt_dog.choices[0]
            message_dict = first_choice.get('message', {}) if isinstance(first_choice, dict) else getattr(first_choice, 'message', {})

        # Extract the 'content' from the 'message' dictionary
        response_content = message_dict.get('content', '') if isinstance(message_dict, dict) else getattr(message_dict, 'content', '')
        
        
       # return jsonify({'prediction': str(predicted_result)})
        return jsonify({'prediction': str(response_content), 'file_path': file_path})

if __name__ == '__main__':
    app.run(debug=True)