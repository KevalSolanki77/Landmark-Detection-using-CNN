# Landmark Detection using CNN

A deep learning project for landmark classification and localization using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras.

## Features

- Landmark image classification
- Bounding box prediction
- TensorFlow/Keras implementation
- CNN-based feature extraction
- IoU (Intersection over Union) evaluation
- Training visualization graphs

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Dataset

if you want to download the data set here is the link and explore :
https://s3.amazonaws.com/google-landmark/metadata/train.csv

CSV file format:
- id
- landmark_id

## Model Architecture

- Conv2D Layers
- MaxPooling Layers
- Dense Layers
- Dual Output:
  - Classification Output
  - Bounding Box Regression Output

## Installation

```bash
git clone https://github.com/yourusername/landmark-detection-cnn.git
cd landmark-detection-cnn
pip install -r requirements.txt
