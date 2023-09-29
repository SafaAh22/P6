# importing librarys
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model


# Load the model from the directory
model = load_model("modelh5.h5")
