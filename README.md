# Image-Oject-Classification  


## Project Description

This project leverages the power of the **MobileNetV2** deep learning model to perform image classification. The model, pre-trained on the **ImageNet** dataset, is capable of classifying images into over 1,000 distinct object categories. By utilizing TensorFlow and Keras, this script allows users to easily input an image file and obtain the predicted class label along with the confidence score. The image is also displayed for visual reference.

The goal of this project is to provide an accessible and easy-to-use image classification tool with a simple interface that can be integrated into various applications or used as a standalone solution for classifying images.

### Key Features:
- **Pre-trained MobileNetV2 Model**: Uses a lightweight yet powerful deep learning architecture pre-trained on ImageNet.
- **Image Classification**: Automatically classifies input images into one of 1,000+ ImageNet categories.
- **Prediction with Confidence**: Displays the predicted label along with a confidence score.
- **Image Visualization**: Displays the input image alongside the predicted label for better understanding.



## Requirements

- **Python**
- **TensorFlowx**
- **Keras**
- **NumPy**
- **Matplotlib**

### Installation

Install the required dependencies using pip:
```bash
pip install tensorflow numpy matplotlib
```



## How to Use

1. Clone the repository (if using Git):
   ```bash
   git clone https://github.com/yourusername/mobilenetv2-image-classifier.git
   cd mobilenetv2-image-classifier
   ```

2. Run the script:
   ```bash
   python classify_image.py
   ```

3. When prompted, enter the path to the image you wish to classify (ensure the image exists at the given path).

4. The model will predict the class of the image, display the image, and output the predicted class along with the confidence score.

### Example:
```bash
Enter the path to the image you want to classify: /path/to/image.jpg
Predicted Class: tabby, tabby cat, tomcat, catamount (98.65%)
```

The image will be shown with the predicted label and the confidence score.



## Code Explanation

- **Model Initialization**: The MobileNetV2 model is loaded with pre-trained weights from ImageNet.
- **Image Processing**: The input image is resized to `224x224` pixels, converted to a NumPy array, and preprocessed to match the input format expected by the model.
- **Prediction**: The image is passed through the model, and the top prediction is decoded into a human-readable label.
- **Error Handling**: Includes handling for file not found errors and any other exceptions that may arise during classification.


