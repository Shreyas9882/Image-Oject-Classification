import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the pre-trained MobileNetV2 model with ImageNet weights
model = MobileNetV2(weights='imagenet')

# Function to classify any input image
def classify_image(image_path):
    try:
        # Load the image and resize it to 224x224 (required for MobileNetV2)
        img = image.load_img(image_path, target_size=(224, 224))

        # Convert the image to a numpy array
        img_array = image.img_to_array(img)

        # Preprocess the image (scale pixel values and adjust dimensions)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class of the input image
        predictions = model.predict(img_array)

        # Decode the predictions into a human-readable label
        decoded_predictions = decode_predictions(predictions, top=1)  # Get the top 1 prediction

        # Extract the most likely prediction
        label, score = decoded_predictions[0][0][1], decoded_predictions[0][0][2]

        # Display the image
        plt.imshow(img)
        plt.axis('off')  # Remove axes for a cleaner display
        plt.title(f"Prediction: {label} ({score * 100:.2f}%)")
        plt.show()

        # Print the single prediction
        print(f"Predicted Class: {label} ({score * 100:.2f}%)")

    except FileNotFoundError:
        print(f"Error: The file at '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Prompt user for the image path
image_path = input("Enter the path to the image you want to classify: ")

# Check if the file exists before classification
if os.path.exists(image_path):
    classify_image(image_path)
else:
    print(f"Error: The file '{image_path}' does not exist.")
