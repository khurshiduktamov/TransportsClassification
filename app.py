import streamlit as st
from fastai.vision.all import *
import plotly.express as px

# Set the page title
st.set_page_config(page_title="Transport Classification Model")

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
    pred, pred_idx, probs = model.predict(img)

    # Expected classes
    expected_classes = ['car', 'boat', 'airplane']

    # Check if the highest probability exceeds the threshold
    if max(probs) >= 0.7:
        # Check if the predicted class is one of the expected classes
        if model.dls.vocab[pred_idx] in expected_classes:
            # Showing the results
            st.success(f'Prediction: {pred}')
            st.info(f'Probability: {probs[pred_idx]*100:.2f}%')

            # Visualisation
            fig = px.bar(x=probs*100, y=model.dls.vocab, labels={'x':'Probability (%)', 'y':'Class'})
            st.plotly_chart(fig)
        else:
            st.error('The uploaded image does not seem to be a car, boat, or airplane.')
    else:
        st.error('The model is not confident enough to make a prediction.')
