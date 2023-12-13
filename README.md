# What-Up-Dog
What Up Dog utilizes multiple machine learning models to not only predict the emotion your dog is experiencing in a picture you upload, but also will explain why your dog could be feeling that emotion based on factors present in the picture. 

## Setup
### Prerequisites 
1. A Python installation
2. An OpenAI API key (requires paid OpenAI API usage credits on your account)

### Local Deployment
To run a local version of our application, follow the steps below:
Note: your working directory while following these steps should be the root directory of this repository

1. Download the required Python packages via:
```
pip install -r requirements.txt
```

2. Obtain an OpenAI API key and store it in a `gpt_api.txt` file

3. Run the application via the following command:
```
python app.py
```
Depending on your Python configuration, you may instead need to run
```
python3 app.py
```

4. The website will be up and running on localhost, and you can upload your dog images to understand your dogs emotions

### How It Works
When you upload an image of your dog (or any image), our application first checks, through a pretrained image captioning model, 
obtains context about what is happening in the image. If the model is unable to find a dog in the image, the application will 
return that a dog was not found. If the image does contain a dog, the image will then be sent to a pretrained image classification model
that we fine-tuned on two different datasets, and classifiy the emotion that the dog is expressing in the image. Both the context 
of the image and the emotion expressed will then be sent to a language model that will return reasoning as to why your dog would be feeling
the predicted emotion given the context of the image.


