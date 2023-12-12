from fastai.vision.all import PILImage

'''
<class 'fastai.learner.Learner'>
<class 'transformers.pipelines.image_to_text.ImageToTextPipeline'>
<class 'openai.OpenAI'>
'''

def preprocess_image(file):
    img = PILImage.create(file)
    img.resize((288,288))
    return img

def predict_emotion(model, img):
    # Make a prediction
    prediction = model.predict(img)
    # Extracting the predicted class
    predicted_result = prediction[0]
    return predicted_result

def get_context(model, img):
    result = model(img)
    string_result = ''.join(map(str,result))
    string_result = string_result[19:]
    return string_result

def dog_precheck(img_to_text_result):
    result = img_to_text_result.lower()
    return result.find('dog') != -1 or result.find('pupp') != -1 

def generate_dog_text(client, emotion, context):
    
    final_string = "You are a dog. If I were to take a picture of you right now you would be {}. Your tone and emotion would be considered {}".format(context,emotion)

    gpt_dog = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": final_string},
            {"role": "user", "content": "Why do you think the dog in the picture is experiencing the emotion we have labeled it with and sent to you?"},
            {"role": "user", "content": "In 1 sentence as if you were a dog: explain why you would be feeling those emotions."}
        ] 
    )
    
    # I beat my head against a wall to parse out the message content, but I used gpt4 to get me to this
    # Check if 'choices' exists in the response and it has at least one element
    if hasattr(gpt_dog, 'choices') and len(gpt_dog.choices) > 0:
        # Extract the 'message' dictionary from the first element of 'choices'
        # This step depends on whether 'choices' is a list of dictionaries or a list of objects
        first_choice = gpt_dog.choices[0]
        message_dict = first_choice.get('message', {}) if isinstance(first_choice, dict) else getattr(first_choice, 'message', {})

    # Extract the 'content' from the 'message' dictionary
    response_content = message_dict.get('content', '') if isinstance(message_dict, dict) else getattr(message_dict, 'content', '')
    return response_content
        