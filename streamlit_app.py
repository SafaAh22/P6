import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import tensorflow
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from io import StringIO

st.write('Projet 6')

# Load the model from the directory
model = load_model("modelh5")


with st.form("my_form"):
   st.write("Upload Dogs Picture")
   uploaded_file = st.file_uploader("Choose a file")