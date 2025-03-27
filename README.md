# 🎥 Computer Vision Lab (CSET340) Assignments

An engaging course repository for computer vision assignments, focusing on practical implementation of image processing, feature detection, object recognition, and deep learning techniques using Python and OpenCV.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Maintenance](https://img.shields.io/badge/Maintenance-Active-brightgreen.svg)

## 📖 Table of Contents
- [Core Components](#-core-components)
- [Technical Requirements](#-technical-requirements)
- [Getting Started](#-getting-started)
- [Assignments](#-assignments)
- [Resources](#-resources)

## 🌟 Core Components

### 📸 Image Processing Fundamentals
- **Basic Operations**
  - Image manipulation
  - Color space conversions
  - Filtering techniques
  - Histogram analysis
- **Advanced Techniques**
  - Edge detection
  - Morphological operations
  - Image enhancement
  - Frequency domain processing

### 🔍 Feature Detection & Recognition
- **Feature Extraction**
  - SIFT/SURF implementations
  - Corner detection
  - Blob detection
  - Template matching
- **Pattern Recognition**
  - Feature matching
  - Object detection
  - Face recognition
  - Scene classification

### 🧠 Deep Learning Approaches
- **Neural Networks**
  - CNN architectures
  - Transfer learning
  - Model training
  - Performance optimization
- **Modern Architectures**
  - ResNet
  - YOLO
  - U-Net
  - Transformers

## 🔧 Technical Requirements

### System Setup
- **Python Environment**
  - Python 3.9+
  - pip or conda
  - Virtual environment
  - Git
- **Required Libraries**
  - OpenCV 4.8+
  - NumPy 1.21+
  - PyTorch 2.0+
  - Matplotlib 3.5+

### Dependencies
```txt
# requirements.txt
opencv-python>=4.8.0
numpy>=1.21.0
torch>=2.0.0
matplotlib>=3.5.0
scikit-image>=0.19.0
pillow>=9.0.0
jupyter>=1.0.0
```

## 🚀 Getting Started

### Setup Instructions
```bash
# Clone the repository
# git clone https://github.com/university/cv-course.git
# cd cv-course

# Create virtual environment
# python -m venv venv
# source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
# pip install -r requirements.txt

# Verify installation
# python -c "import cv2; print(cv2.__version__)"
```


## ⭐ Grading

### Evaluation Criteria
| Component | Weight | Description |
|-----------|---------|------------|
| Implementation | Code correctness & efficiency |

| Results | Output quality & analysis |

## 🤝 Contributing

### Guidelines
1. Follow PEP 8 style guide
2. Document all functions
3. Include unit tests
4. Maintain clean commit history

## 🚀 Assignments
## Experiment 1
1. Task-1: - Perform following operations on image-
 
1.1 Image Resizing: Resizing involves changing the dimensions of an image, either by scaling it up or down. 
1.1	Image resizing (interpolation methods)
1.1.1	Linear
1.1.2	Nearest Neighbors
1.1.3	Polynomial 
1.2 Image Blurring: Blurring is used to reduce image detail, suppress noise, or create artistic effects. Common techniques include:
1.2	Image blurring
1.2.1	Box blurring
1.2.2	Gaussian blurring
1.2.3	Adaptive blurring

Task-2: - Apply Machine Learning Algorithm and find the model accuracy based on K fold Cross Validation with (80-20 train-test split).  
2.1	Use MNIST dataset
2.2	Use any two of the following algorithms-
2.2.1	Naive Bayesian or its variant.
2.2.2	Support Vector Machine (SVM) or its variant
2.2.3	Decision Trees/ Random Forest.
2.2.4	AdaBoost or other ensemble algorithms.
2.2.5	Artificial Neural Networks (NN) or its variant.
2.3	Results should be obtained on following parameters-
2.3.1	Accuracy
2.3.2	Precision (Positive Predictive Value)
2.3.3	Recall (Sensitivity)
2.3.4	F-Measure
2.3.5	Confusion Matrix
2.3.6	ROC
2.3.7	AUC
Appendix:- 
About MNIST :- 
•	The MNIST dataset stands for "Modified National Institute of Standards and Technology". 
•	The dataset contains a large collection of handwritten digits that is commonly used for training various image processing systems. 
•	 
•	The dataset was created by re-mixing samples from NIST's original datasets, which were taken from American Census Bureau employees and high school students. 
•	It contains 60,000 training images and 10,000 testing images, each of which is a grayscale image of size 28x28 pixels.
o	Number of Instances: 70,000 images
o	Number of Attributes: 784 (28x28 pixels)
o	Target: Column represents the digit (0-9) corresponding to the handwritten image
o	Pixel 1-784: Each pixel value (0-255) represents the grayscale intensity of the corresponding pixel in the image.
o	The dataset is divided into two main subsets:
	Training Set: Consists of 60,000 images along with their labels, commonly used for training machine learning models.
	Test Set: Contains 10,000 images with their corresponding labels, used for evaluating the performance of trained models.
•	Link:- https://www.kaggle.com/datasets/hojjatk/mnist-dataset 
•	Note:- Use sklearn, pyspark, or any other ML library for applying the ML algorithms.
o	Load the dataset in sklearn using ‘load_digits’.
o	Load the dataset in pyspark using 'spark.read.csv()”


---

*Last Updated: February 2025*
