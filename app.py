import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('fruit_classifier_model.h5')

# Define class labels
class_names = ['apple', 'banana', 'cherry', 'grapes', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberry', 'watermelon']

# Streamlit App Layout
st.title("Fruit Classifier")
st.write("Upload an image of a fruit, and this model will classify it!")

# Upload image
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

# Define thresholds
CONFIDENCE_THRESHOLD = 0.5
MAX_PREDICTIONS = 5  # Number of top predictions to check

if uploaded_file is not None:
    try:
        # Open and display the image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = img.convert('RGB')  # Convert to RGB to avoid alpha channel issues
        img = img.resize((224, 224))  # Resize to match model's expected input size
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        prediction = model.predict(img_array)[0]

        # Sort predictions in descending order
        top_indices = prediction.argsort()[-MAX_PREDICTIONS:][::-1]
        top_predictions = [(class_names[idx], prediction[idx]) for idx in top_indices]

        # Check if any top predictions meet the confidence threshold
        confident_predictions = [
            (fruit, conf) for fruit, conf in top_predictions if conf >= CONFIDENCE_THRESHOLD
        ]

        if not confident_predictions:
            st.write("Unknown Fruit: The image does not match any known fruit classes.")
        else:
            # Display the top confident predictions
            for fruit, confidence in confident_predictions:
                st.write(f"Predicted Fruit: **{fruit}**")
                st.write(f"Confidence: **{confidence * 100:.2f}%**")

    except Exception as e:
        st.error(f"Error processing image: {e}")