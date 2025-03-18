import streamlit as st
import tensorflow as tf
from PIL import Image
from model import model
import cv2
import numpy as np
import joblib
import os

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

model.load_weights('BrDX.keras')


st.title("BrDX",anchor=None)
st.header("Brain Tumor Detection")
st.divider()
st.text("Upload MRI Image")

label_encoder=joblib.load('le.pkl')

uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"],accept_multiple_files=False)





if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    image_resized = cv2.resize(image,(64,64))/255
    
    predicted = model.predict(np.array([image_resized]))
    predicted_class_index = np.argmax(predicted,axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_class_index)
    predicted_prob = predicted[0][predicted_class_index]
    
    
    if st.button("Detect",use_container_width=True):
        st.success(f"The uploaded image is {predicted_class[0]} with {int(predicted_prob[0]*100)} %  prediction accuracy")
        

        
        g='''if st.button("Locate",use_container_width=True):
            st.info("Locating")'''
            
