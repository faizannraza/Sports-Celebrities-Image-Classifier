Sports Celebrities Image Classifier
This project is designed to classify sports celebrities based on images provided by users. The model is built using Python, with image preprocessing and classification implemented using various techniques, including Support Vector Machines (SVM). The project also involves data preprocessing, feature engineering, and deployment on Amazon EC2.

Project Overview
Objective:
Classify images of sports celebrities including Virat Kohli, Roger Federer, Serena Williams, Maria Sharapova, and Lionel Messi.

Technologies Used:

Programming Languages: Python, JavaScript
Libraries: OpenCV, scikit-learn, PyWavelets, NumPy, Matplotlib
Deployment: Amazon EC2
Development Environment: PyCharm
Dataset
The dataset consists of images of the following sports celebrities:

Virat Kohli
Roger Federer
Serena Williams
Maria Sharapova
Lionel Messi
Data Preprocessing and Feature Engineering
Image Reading and Conversion:

Images are read using OpenCV and converted to grayscale for processing.
Face Detection:

Utilized Haar Cascades for detecting faces and eyes in images.
Cropped images where faces with at least two eyes are detected.
Image Cropping:

Images are cropped to include only faces with two eyes to standardize the dataset.
Wavelet Transform:

Applied Wavelet Transform (using PyWavelets) to enhance feature extraction by removing low-frequency content.
Feature Vector Creation:

Extracted features from images by combining raw pixel values with wavelet coefficients.
Resized images to 32x32 pixels and flattened them into feature vectors.
Model Training
Support Vector Machine (SVM):

An SVM model with an RBF kernel is trained on the feature vectors.
Accuracy achieved: ~79% on test data.
Hyperparameter Tuning:

Used GridSearchCV to optimize model parameters for SVM, RandomForestClassifier, and LogisticRegression.
Best model and parameters are selected based on cross-validation results.
Deployment
The application is deployed on Amazon EC2, where users can upload images and receive predictions on the sports celebrity depicted in the image.

Evaluation
SVM with RBF Kernel: Achieved an accuracy of approximately 79% on test data.
Model Tuning: Various models and hyperparameters were evaluated using GridSearchCV.

Installation and Usage
Clone the Repository: git clone https://github.com/yourusername/sports-celebrities-image-classifier.git
