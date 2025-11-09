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

```
PlantDL/
│
├── app.py                      # Streamlit Web App
├── plant_disease_model.h5      # Trained Model File
├── class_labels.json           # Class label mappings
├── dataset/                    # PlantVillage dataset (Tomato, Potato, Pepper)
├── README.md                   # Project documentation
└── requirements.txt            # Required Python packages
```

---

## 🧰 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow
streamlit
numpy
pillow
json5
```

---

## 🚀 How to Run

1. Clone or download this project.
2. Place your trained model (`plant_disease_model.h5`) and labels file (`class_labels.json`) in the same folder as `app.py`.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Upload a leaf image (JPG/PNG) from the dataset to test.

---

## 💡 Example Output

```
🌿 Predicted Disease: Tomato___Early_blight
Confidence: 96.34%
🧫 Cause: Fungus Alternaria solani
⚕️ Symptoms: Brown concentric rings on lower leaves, yellowing, and defoliation.
💊 Treatment: Remove infected debris, rotate crops, and apply preventive fungicides.
```

---

## 📈 Model Performance

| Metric | Value |
|---------|-------|
| Training Accuracy | 92.6% |
| Validation Accuracy | 91.3% |
| Loss | 0.28 |

---

## 🌱 Future Enhancements

- Add more crop types (Corn, Apple, Grape).
- Include real-time webcam-based disease detection.
- Integrate mobile app interface for field diagnosis.
- Use transfer learning with Vision Transformers for improved accuracy.

---

## 📚 References

- [PlantVillage Dataset - Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- TensorFlow Documentation  
- Streamlit Documentation  
- “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” *Google Research, 2018*

---

## © License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.
