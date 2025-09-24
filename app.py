import streamlit as st
from PIL import Image
from model import model
from unet_model import unet
import cv2
import numpy as np
import joblib
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
st.title("BrDX")
st.header("Brain Tumor Classification and Segmentation")
st.divider()

unet.load_weights('models/unet_model.h5')
model.load_weights('models/cnn_model.keras')
label_encoder = joblib.load('le.pkl')

st.text("Upload MRI Image")
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_image:
    if "uploaded_image_name" not in st.session_state or st.session_state.uploaded_image_name != uploaded_image.name:
        st.session_state.uploaded_image_name = uploaded_image.name
        st.session_state.predicted_class = None
        st.session_state.predicted_prob = None
        st.session_state.final_image = None
        st.session_state.mask_image = None
        st.session_state.y_pred = None
        st.session_state.locate_clicked = False

def pre_processing(image_obj):
    resized_image = cv2.resize(image_obj, (256, 256))
    x = resized_image / 255.0
    x = np.expand_dims(x, axis=0)
    y_pred = unet.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = (y_pred >= 0.5).astype(np.uint8)
    return y_pred

def image_postprocess(y_pred):
    mask = (y_pred * 255).astype(np.uint8)
    return mask

if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption="Preview of Uploaded Image", width=250)
    
    if st.button("Classify", use_container_width=True):
        st.session_state.locate_clicked = False
        
        image_resized = cv2.resize(image, (64, 64)) / 255.0
        predicted = model.predict(np.array([image_resized]))
        predicted_class_index = np.argmax(predicted, axis=1)

        predicted_class = label_encoder.inverse_transform(predicted_class_index)
        predicted_prob = predicted[0][predicted_class_index][0]
        st.session_state.predicted_class = predicted_class[0]
        st.session_state.predicted_prob = int(predicted_prob * 100)

        image_with_bbox = image.copy()

        if st.session_state.predicted_class != "no_tumor":
            y_pred = pre_processing(image)
            st.session_state.y_pred = y_pred
            mask = image_postprocess(y_pred)
            st.session_state.mask_image = Image.fromarray(mask)
            
            if np.sum(y_pred) > 0:
                contours, _ = cv2.findContours(y_pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    margin_fraction = 0.3
                    margin_x = int(w * margin_fraction)
                    margin_y = int(h * margin_fraction)
                    x = max(0, x - margin_x)
                    y = max(0, y - margin_y)
                    w = w + 2 * margin_x
                    h = h + 2 * margin_y
                    
                    orig_h, orig_w = image.shape[:2]
                    scale_x = orig_w / 256
                    scale_y = orig_h / 256
                    x_orig = int(x * scale_x)
                    y_orig = int(y * scale_y)
                    w_orig = int(w * scale_x)
                    h_orig = int(h * scale_y)
                    
                    x_orig = max(0, x_orig)
                    y_orig = max(0, y_orig)
                    if x_orig + w_orig > orig_w:
                        w_orig = orig_w - x_orig
                    if y_orig + h_orig > orig_h:
                        h_orig = orig_h - y_orig
                    
                    cv2.rectangle(image_with_bbox, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 0, 255), 2)
                    label_text = st.session_state.predicted_class.upper()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = orig_w / 450.0
                    thickness = max(1, int(font_scale * 2))
                    text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    text_x = x_orig
                    text_y = y_orig - 10 if y_orig - 10 > text_size[1] else y_orig + text_size[1] + 10
                    cv2.putText(image_with_bbox, label_text, (text_x, text_y), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
        else:
            st.session_state.mask_image = None
        
        st.session_state.final_image = Image.fromarray(image_with_bbox)
        st.session_state.locate_clicked = True

if st.session_state.get("locate_clicked") and st.session_state.final_image is not None:
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.final_image, caption="Uploaded Image with Bounding Box", use_container_width=True)
        if st.session_state.mask_image is not None:
            with col2:
                st.image(st.session_state.mask_image, caption="Segmentation Mask", use_container_width=True)
    st.success(f"The uploaded image is {st.session_state.predicted_class.upper()} with {st.session_state.predicted_prob}% prediction confidence")