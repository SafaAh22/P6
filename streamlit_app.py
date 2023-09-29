import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import tensorflow
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

# Load the model from the directory
model = load_model("modelh5.h5")