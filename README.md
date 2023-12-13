# What-Up-Dog


What Up Dog utilizes multiple machine learning models to not only predict the emotion your dog is experiencing in a picture you upload, but also will explain why your dog could be feeling that emotion based on factors present in the picture.

## Screenshots

![App Screenshot](https://cdn.discordapp.com/attachments/1158126522427785276/1184224840446660670/image.png?ex=658b3253&is=6578bd53&hm=a17fa85cfecdefc997f9ee0d4c96a12552aca5e3d37090bb1dbb44f7b0ef4246&)


## Setup
###preprequisties

1. A python installation
2. An opnnAI API key ( requres paid opnnAI API usage credits on your account)


## Installation

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

## Deployment

To run a local version of our application, follow the steps below:
Note: your working directory while following these steps should be the root directory of this repository

### How It Works
When you upload an image of your dog (or any image), our application first checks, through a pretrained image captioning model, 
obtains context about what is happening in the image. If the model is unable to find a dog in the image, the application will 
return that a dog was not found. If the image does contain a dog, the image will then be sent to a pretrained image classification model
that we fine-tuned on two different datasets, and classifiy the emotion that the dog is expressing in the image. Both the context 
of the image and the emotion expressed will then be sent to a language model that will return reasoning as to why your dog would be feeling
the predicted emotion given the context of the image.
## API Reference

#### Get all items

```http
  GET /api/items
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |

#### Get item

```http
  GET /api/items/${id}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


## Acknowledgements

 - [Dog Emotions prediction](https://www.kaggle.com/datasets/devzohaib/dog-emotions-prediction/data)
 - [Dog Emotion](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion)
 - [Dog emotion classifier](https://www.kaggle.com/code/karolgu/dog-emotion-classifier)

## Appendix

Any additional information goes here


## Documentation

[Documentation](https://linktodocumentation)


## Authors

- [theUncluded](https://github.com/theUncluded)
- [Trollicorn](https://github.com/Trollicorn)
- [mkim2780](https://github.com/mkim2780)

