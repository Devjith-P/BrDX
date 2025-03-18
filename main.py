import streamlit as st  # type: ignore
import tensorflow as tf
from PIL import Image
from model import model
from unet_model import unet
import cv2
import numpy as np
import joblib
import os

# Disable unnecessary TensorFlow optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load models
model.load_weights('BrDX.keras')
unet.load_weights('model.h5')


st.title("BrDX")
st.header("Brain Tumor Detection")
st.divider()
st.text("Upload MRI Image")

col1 , col2 = st.columns(2)

label_encoder = joblib.load('le.pkl')

if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None
    st.session_state.predicted_prob = None
    st.session_state.final_image = None
    st.session_state.locate_clicked = False

st.session_state.uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

def pre_processing(image_obj):
    image = cv2.resize(image_obj, (256, 256))  
    x = image / 255.0  
    x = np.expand_dims(x, axis=0)  
    y_pred = unet.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = (y_pred >= 0.5).astype(np.uint8)
    return y_pred

def image_postprocess(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255
    mask = y_pred[:, :, -1]  
    return mask

def image_extract_segment(image_obj, mask):
    image = cv2.resize(image_obj, (256, 256))
    image_masked = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    return image_masked

def draw_bounding_box(image, mask, color=(0, 255, 0)):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:  
                cv2.rectangle(image, (x-20, y-20), (x + w-10, y + h-10), color, 2)
                break
    return image

if st.session_state.uploaded_image:
    image = np.array(Image.open(st.session_state.uploaded_image))
    image_resized = cv2.resize(image, (64, 64)) / 255  
    predicted = model.predict(np.array([image_resized]))  
    predicted_class_index = np.argmax(predicted, axis=1)  
    predicted_class = label_encoder.inverse_transform(predicted_class_index)  
    predicted_prob = predicted[0][predicted_class_index][0]  
    st.session_state.predicted_class = predicted_class[0]
    st.session_state.predicted_prob = int(predicted_prob * 100)
    y_pred = pre_processing(image)
    mask = image_postprocess(y_pred)
    image_masked = image_extract_segment(image, mask)
    st.session_state.final_image = Image.fromarray(image_masked) 
    
    if st.button("Detect", use_container_width=True):
        if st.session_state.predicted_class == 'no_tumor':
            st.success(f"Image has NO TUMOR with {st.session_state.predicted_prob}% prediction accuracy")
        else:
            st.success(f"The uploaded image is {st.session_state.predicted_class.upper()} with {st.session_state.predicted_prob}% prediction accuracy")
        st.session_state.locate_clicked = True

if st.session_state.locate_clicked and st.session_state.final_image:
    if st.button("Locate", use_container_width=True):
        with col1:
            st.image(st.session_state.uploaded_image, caption="Uploaded Tumor Image", use_container_width=True)
        with col2:
            st.image(st.session_state.final_image, caption="Segmented Tumor Area", use_container_width=True)
