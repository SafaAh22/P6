import streamlit as st
import numpy as np
import gdown
import joblib
import pandas as pd
import sklearn
import tensorflow
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import StringIO
import requests

st.write('Projet 6')

# Load the model from the directory
# model = load_model("finalmodelh5")
# URL to the Dropbox direct download link
dropbox_url = 'https://www.dropbox.com/scl/fi/f7du9t15qxqz05aajihoi/InceptionV3modelh5.h5?rlkey=oa85yr9wzpjhy8vi4bpxxzht3&st=3j6ek61h&dl=1'
model_path = 'InceptionV3modelh5.h5'

# Download the file from Dropbox
with requests.get(dropbox_url, stream=True) as response:
    with open(model_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Load the model
model = load_model(model_path)

# Load the model
model = load_model(model_path)

breed_names = joblib.load('breed_names.joblib')

# Define a function to load and preprocess image
def load_and_preprocess_image(uploaded_file):
    # Load image and set the size to (224, 224)
    img = Image.open(uploaded_file).resize((224, 224))
    
    # Convert image to array and expand dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array /= 255.

    return img_array


with st.form("my_form"):
   st.write("Upload Dogs Picture")
   uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
      if uploaded_file is not None:
        img_array = load_and_preprocess_image(uploaded_file)
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_breed = breed_names[predicted_class]
        st.write(f"Predicted breed: {predicted_breed}")
        st.image(uploaded_file)
      else:
        st.write("Please upload an image.")
