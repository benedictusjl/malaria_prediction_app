import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


st.write("""
          # Malaria Image Classification
          """
          )
upload_file = st.sidebar.file_uploader("Upload Cell Images", type="png")
Generate_pred=st.sidebar.button("Predict")
#model=tf.keras.models.load_model('efficientnet_malaria_prediction.h5',custom_objects={'KerasLayer':hub.KerasLayer})
model=tf.keras.models.load_model('efficientnet_model.h5')

def import_n_pred(image_data, model):
    size = (96,96)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred

if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('Cell Image', expanded = True):
        st.image(image, use_column_width=True)
    pred=import_n_pred(image, model)
    labels = ['Parasitized', 'Uninfected']
    st.title("Prediction of image is {}".format(labels[np.argmax(pred)]))