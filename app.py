import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Title
st.title('Transport classification model')

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

    # Showing the results
    st.success(f'Prediction: {pred}')
    st.info(f'Probability: {probs[pred_id]*100:.2f}%')

    # Visualisation
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)