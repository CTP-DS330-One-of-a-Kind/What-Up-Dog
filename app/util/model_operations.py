from fastai.vision.all import PILImage

def preprocess_image(file):
    """
    Preprocess image file into an image format to be used by machine learning models
    """
    img = PILImage.create(file)
    img.resize((288,288))
    return img

def predict_emotion(model, img):
    """
    Predicts emotion of given image with given machine learning model 
    """
    # Make a prediction
    prediction = model.predict(img)
    # Extracting the predicted class
    predicted_result = prediction[0]
    return predicted_result

def get_context(model, img):
    """
    Gets context of given image with given image-to-text model
    """
    # Do Image to Text
    text_from_image = model(img)
    # Extract results
    string_result = ''.join(map(str,text_from_image))
    string_result = string_result[19:]
    return string_result

def dog_precheck(img_to_text_result):
    """
    Returns true if the given image to text results are about dogs or puppies
    """
    result = img_to_text_result.lower()
    return result.find('dog') != -1 or result.find('pupp') != -1 

def generate_dog_text(client, emotion, context):
    """
    Connects to OpenAI's GPT 3.5 model and generates explanation of why dog is feeling given emotion in given context 
    """
    final_string = "You are a dog. If I were to take a picture of you right now you would be {}. Your tone and emotion would be considered {}".format(context,emotion)

    gpt_dog = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": final_string},
            {"role": "user", "content": "Why do you think the dog in the picture is experiencing the emotion we have labeled it with and sent to you?"},
            {"role": "user", "content": "In 1 sentence as if you were a dog: express this explain why you would be feeling those emotions in the first person, as if you're experiencing it. Do not say 'as a dog...'"}
        ] 
    )
    # Check if 'choices' exists in the response and it has at least one element
    if hasattr(gpt_dog, 'choices') and len(gpt_dog.choices) > 0:
        # Extract the 'message' dictionary from the first element of 'choices'
        # This step depends on whether 'choices' is a list of dictionaries or a list of objects
        first_choice = gpt_dog.choices[0]
        message_dict = first_choice.get('message', {}) if isinstance(first_choice, dict) else getattr(first_choice, 'message', {})

    # Extract the 'content' from the 'message' dictionary
    response_content = message_dict.get('content', '') if isinstance(message_dict, dict) else getattr(message_dict, 'content', '')
    return response_content
        