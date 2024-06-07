import streamlit as st
from fastai.vision.all import *
import plotly.express as px
from pathlib import Path
import pathlib
import platform

# Platform-specific adjustments
plt = platform.system()
pathlib.WindowsPath = pathlib.PosixPath

# Set the page title
st.set_page_config(page_title="AQI Predictions")

# Title
st.title('Transport Classification Model')

st.write("It works only with Cars, Airplanes, and Boats for now")

# Uploading a pic
file = st.file_uploader('Upload a photo', type=['jpg', 'jpeg', 'png', 'gif', 'svg'])

if file:
    # Showing loaded pic
    st.image(file)

    # PIL image converter
    img = PILImage.create(file)

    # Loading the model
    model = load_learner('transport_model.pkl')

    # Prediction
    pred, pred_id, probs = model.predict(img)

    # Expected classes
    expected_classes = ['car', 'boat', 'airplane']

    # Showing the results
    if pred in expected_classes:
        st.success(f'Prediction: {pred}')
        st.info(f'Probability: {probs[pred_id]*100:.2f}%')

        # Visualisation
        fig = px.bar(x=probs*100, y=model.dls.vocab, labels={'x':'Probability (%)', 'y':'Class'})
        st.plotly_chart(fig)
    else:
        st.error('The uploaded image is not a car, boat, or airplane.')
