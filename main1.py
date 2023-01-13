
import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from keras.models import model_from_json

def load_model():
    # Function to load and return neural network model 
    json_file = open('Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("my_model_weight.h5")
    return loaded_model

def create_img(path):
    #Function to load,normalize and return image 
    print(path)
    im = Image.open(path).convert('RGB')
    
    im = np.array(im)
    
    im = im/255.0
    
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225


    im = np.expand_dims(im,axis  = 0)
    return im
def predict(path):
    #Function to load image,predict heat map, generate count and return (count , image , heat map)
    model = load_model()
    image = create_img(path)
    ans = model.predict(image)
    count = np.sum(ans)
    return count

import numpy as np




app = FastAPI()

class UserInput(BaseModel):
    user_input: str

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.get('/predict/')
async def predict(UserInput: UserInput):

    prediction = predict([UserInput.user_input])

    return {"prediction": float(prediction)}
@app.put("/items/{item_id}")
def update_item(user_input: UserInput, ):
    prediction = predict([user_input.user_input])

    return {"prediction": float(prediction)}
