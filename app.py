import streamlit as st
import torch
from PIL import Image
import json

# Import necessary funcitons from predict.py
from predict import load_checkpoint_model, predict, process_image

st.title("Welcome to the Flower Classification App!")
st.write("Please upload a flower image, and the app will predict its class of flower.")

uploaded_file = st.file_uploader(
    "Choose a flower image...", type=[".jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load the model and category names for each prediction
    model = load_checkpoint_model("checkpoint.pth", device="cpu")
    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)

    image = Image.open(uploaded_file)
    # Process the image
    image_tensor = process_image(uploaded_file)
    st.write(f"Preprocessed Image Tensor: {image_tensor}")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Predicting...")

    try:
        # Predict the class of the flower
        probs, classes = predict(image, model, top_k=5, device="cpu")

        class_names = [cat_to_name[cls] for cls in classes]

        # Display the predictions
        for p, cls in zip(probs, class_names):
            st.write(f"{cls}: {p:.3f}")
    except Exception as e:
        st.error(f"An error occured: {e}")
