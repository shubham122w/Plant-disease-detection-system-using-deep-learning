# 🌿 Plant Disease Detection using Deep Learning

This project uses a Convolutional Neural Network (CNN) model trained on the **PlantVillage Dataset** to automatically detect and classify plant leaf diseases.  
The system helps farmers and researchers identify diseases early, improving crop yield and reducing pesticide misuse.

---

## 📸 Features

- 🌱 Detects common diseases in **Tomato, Potato, and Pepper** plants.  
- 🧠 Built using **Deep Learning (MobileNetV2)**.  
- ⚡ Real-time detection via a **Streamlit web app**.  
- 📊 Displays **prediction confidence**.  
- 💊 Provides **disease cause, symptoms, and treatment** suggestions.  

---

## 🧩 Dataset

- Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Classes used in this project:

| Crop | Disease / Condition |
|-------|--------------------|
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |

---

## 🧠 Model Architecture

- Base Model: **MobileNetV2** (Pretrained on ImageNet)
- Layers Added:
  - GlobalAveragePooling2D
  - Dense (ReLU activation)
  - Dropout (to prevent overfitting)
  - Dense (Softmax output layer)
- Optimizer: `adam`
- Loss Function: `categorical_crossentropy`
- Evaluation Metric: `accuracy`

---

## ⚙️ Project Structure

# Plant-disease-detection-system-using-deep-learning
