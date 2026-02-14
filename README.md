# 🌿 Plant Disease Detection using CNN (Deep Learning Project)

<p align="center">
  <img src="https://img.shields.io/badge/Python-DeepLearning-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-CNN-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/ComputerVision-Image%20Classification-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Project-DeepLearning%20Skills-black?style=for-the-badge">
</p>

<p align="center">
🚀 A deep learning based **Plant Disease Detection System** built using Convolutional Neural Networks (CNN).  
This project demonstrates real-world computer vision skills — from dataset preprocessing and model training to deployment-ready prediction logic.
</p>

---

## 📌 Table of Contents

* [✨ Project Overview](#-project-overview)
* [🧠 Model Architecture](#-model-architecture)
* [📂 Project Structure](#-project-structure)
* [⚙️ Tech Stack](#️-tech-stack)
* [🚀 Features](#-features)
* [📊 Dataset](#-dataset)
* [🧪 Model Training](#-model-training)
* [🔮 Prediction System](#-prediction-system)
* [📸 Sample Workflow](#-sample-workflow)
* [🧑‍💻 Skills Demonstrated](#-skills-demonstrated)
* [⚡ Installation](#️-installation)
* [▶️ How to Run](#️-how-to-run)
* [📬 Author](#-author)

---

# ✨ Project Overview

This project uses a **Convolutional Neural Network (CNN)** to classify plant leaf images into multiple disease categories using the PlantVillage dataset.

The system:

✅ Detects plant diseases from leaf images
✅ Performs multi-class classification
✅ Suggests remedies for detected diseases
✅ Demonstrates deep learning + computer vision pipeline

---

# 🧠 Model Architecture

The CNN architecture includes:

* Conv2D layers for feature extraction
* MaxPooling layers for dimensionality reduction
* Fully Connected Dense layers
* Dropout layers to prevent overfitting
* Softmax output layer for multi-class classification

Pipeline:

```
Image → CNN Feature Extraction → Dense Layers → Disease Prediction
```

---

# 📂 Project Structure

```
PLANT DISEASE PREDICTOR
│
├── plantvillage dataset/
│
├── app.py                # Prediction / Streamlit logic
├── plant.ipynb           # Model training notebook
├── class_indices.json    # Class label mapping
├── plant.json            # Additional configs
└── README.md
```

---

# ⚙️ Tech Stack

| Technology         | Purpose            |
| ------------------ | ------------------ |
| Python             | Core Programming   |
| TensorFlow / Keras | CNN Model Training |
| NumPy              | Data Processing    |
| PIL                | Image Handling     |
| Matplotlib         | Visualization      |
| Streamlit          | Deployment UI      |

---

# 🚀 Features

* 🧠 Deep Learning CNN architecture
* 🌱 Multi-class plant disease classification
* 📸 Image upload prediction system
* 🩺 Remedy suggestion system
* ⚡ Clean deployment-ready structure
* 📊 Training visualization graphs

---

# 📊 Dataset

Dataset Used: **PlantVillage Dataset**

Contains:

* Multiple crops
* Healthy and diseased leaf images
* Color, grayscale and segmented variants

Classes include:

```
Apple___Apple_scab
Tomato___Early_blight
Grape___Black_rot
Corn___Leaf_Blight
...and many more
```

---

# 🧪 Model Training

Training pipeline includes:

* ImageDataGenerator preprocessing
* Validation split
* CNN training with Adam optimizer
* Accuracy & loss monitoring
* Model evaluation

Key Parameters:

```
Image Size: 224x224
Batch Size: 32
Epochs: 5
Loss: categorical_crossentropy
```

---

# 🔮 Prediction System

The prediction pipeline:

1️⃣ Upload plant leaf image
2️⃣ Image preprocessing & normalization
3️⃣ CNN prediction
4️⃣ Disease classification
5️⃣ Remedy recommendation

Outputs:

```
Predicted Disease
Confidence Score
Suggested Remedy
```

---

# 📸 Sample Workflow

```
Upload Image → CNN Model → Predicted Disease → Remedy Output
```

---

# 🧑‍💻 Skills Demonstrated

This project highlights:

* Deep Learning (CNN)
* Computer Vision
* Image Preprocessing
* Multi-class Classification
* TensorFlow/Keras Model Design
* Dataset Engineering
* Deployment-ready AI pipeline

Also complements strong background in:

✔ Machine Learning
✔ Generative AI
✔ Agentic AI
✔ Data Structures & Algorithms (SDE Side)

---

# ⚡ Installation

Clone the repository:

```
git clone https://github.com/yourusername/Plant-Disease-Detection-CNN.git
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ How to Run

If using Streamlit UI:

```
streamlit run app.py
```

Or run prediction logic directly from notebook.

---

# 📬 Author

**Vashishtha Verma**

AI / ML Engineer | Deep Learning Enthusiast 

* Machine Learning & Deep Learning
* Full-Stack AI Projects
* Strong DSA & SDE Foundations


