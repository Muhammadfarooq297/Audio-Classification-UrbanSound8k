# UrbanSound8K Audio Classification using ANN

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)
![MFCC](https://img.shields.io/badge/Audio%20Features-MFCC-orange)


This project focuses on classifying urban sounds using the **UrbanSound8K Dataset** and an **Artificial Neural Network (ANN)**. The model is designed to predict sound categories using  **MFCC (Mel-Frequency Cepstral Coefficients)** features extracted from audio clips.


## 🚀 Overview

The **UrbanSound8K** dataset contains 8732 sound excerpts (<= 4s) from 10 different urban sound classes. This project leverages **Deep Learning** techniques to classify these sounds based on MFCC features with high accuracy.

## 📊 Dataset

- **UrbanSound8K Dataset**: Contains 8732 sound excerpts (<=4s) of urban sounds classified into the following 10 categories:
  - Air Conditioner
  - Car Horn
  - Children Playing
  - Dog Bark
  - Drilling
  - Engine Idling
  - Gun Shot
  - Jackhammer
  - Siren
  - Street Music

The dataset is preprocessed using **MFCC** to extract relevant features for training the neural network.

## 🧠 Model Architecture

The model uses **Artificial Neural Network (ANN)** with the following layers:

- **Input Layer**: Accepts MFCC features from the UrbanSound8K dataset.
- **Hidden Layers**: Dense layers with **ReLU** activation functions.
- **Optimizers**: Model uses adam optimizer to update weights.
- **Output Layer**: Uses **Softmax** for classification into 10 urban sound categories.
  
### Key Features:
- **MFCC Feature Extraction**: Extracts important features from the audio data.
- **Artificial Neural Network (ANN)**: Utilized for classifying urban sounds.
- **Hyperparameter Tuning**: Applied to improve model performance.


## 📈 Results

- **Accuracy**: The model achieved high accuracy of 85% in classifying urban sound categories.
- **Confusion Matrix**: Displays the model's performance across different sound classes, providing insights into how well the model differentiates between similar sounds.

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Libraries**: 
  - TensorFlow/Keras
  - Librosa
  - Numpy, Pandas, Scikit-learn
  - Matplotlib for visualizations


## 🏃‍♂️ How to Run the Project

- **Clone the repository**:
   ```bash
   git clone https://github.com/Muhammadfarooq297/Audio-Classification-UrbanSound8k
   cd Audio-classification-UrbanSound8k



## Conclusion
This project showcases the effective use of MFCC and Artificial Neural Networks to classify urban sounds with high accuracy. By training on diverse sound classes from the UrbanSound8K dataset, the model demonstrates the potential of deep learning in environmental sound recognition. With further tuning and enhancement, this work can serve as a foundation for real-world audio classification applications such as smart city noise monitoring or sound-based event detection.