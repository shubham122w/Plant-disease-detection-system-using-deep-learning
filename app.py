import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
from PIL import Image

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.h5")
    return model

model = load_model()

# ---------------------- LOAD LABELS ----------------------
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)
class_labels = {v: k for k, v in class_labels.items()}

# ---------------------- PLANT DISEASE INFO ----------------------
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "cause": "Bacterium *Xanthomonas campestris pv. vesicatoria*",
        "symptoms": "Small, dark, water-soaked spots on leaves which enlarge and cause defoliation.",
        "treatment": "Avoid overhead irrigation, use certified seeds, and apply copper-based bactericides."
    },
    "Pepper__bell___healthy": {
        "cause": "None – healthy plant",
        "symptoms": "Leaves are green, smooth, and free from spots or yellow patches.",
        "treatment": "Maintain balanced watering, ensure good air circulation, and check regularly for pests."
    },
    "Potato___Early_blight": {
        "cause": "Fungus *Alternaria solani*",
        "symptoms": "Dark concentric rings appear on older leaves, leading to yellowing and withering.",
        "treatment": "Apply fungicides like Mancozeb or Chlorothalonil and rotate crops annually."
    },
    "Potato___Late_blight": {
        "cause": "Oomycete *Phytophthora infestans*",
        "symptoms": "Dark brown patches on leaves with white mold underneath, rapid leaf decay.",
        "treatment": "Remove affected leaves immediately and apply copper-based fungicides."
    },
    "Potato___healthy": {
        "cause": "None – healthy plant",
        "symptoms": "Uniform green color, firm leaves without any patches or mold.",
        "treatment": "Maintain proper watering, nutrient levels, and pest control."
    },
    "Tomato___Bacterial_spot": {
        "cause": "Bacterium *Xanthomonas vesicatoria*",
        "symptoms": "Small black spots on leaves and fruits, often with yellow halos.",
        "treatment": "Use disease-free seeds, avoid handling wet plants, and use copper-based sprays."
    },
    "Tomato___Early_blight": {
        "cause": "Fungus *Alternaria solani*",
        "symptoms": "Brown concentric rings on lower leaves, yellowing, and defoliation.",
        "treatment": "Remove infected debris, rotate crops, and apply preventive fungicides."
    },
    "Tomato___Late_blight": {
        "cause": "Pathogen *Phytophthora infestans*",
        "symptoms": "Dark, water-soaked lesions on leaves and stems, often spreading quickly.",
        "treatment": "Destroy infected plants and apply metalaxyl-based fungicides promptly."
    },
    "Tomato___Leaf_Mold": {
        "cause": "Fungus *Passalora fulva*",
        "symptoms": "Yellow patches on upper leaf surfaces and olive-green mold underneath.",
        "treatment": "Improve ventilation, avoid overhead watering, and apply sulfur-based fungicides."
    },
    "Tomato___healthy": {
        "cause": "None – healthy plant",
        "symptoms": "Leaves are smooth and vibrant green without discoloration or mold.",
        "treatment": "Continue normal watering and sunlight exposure; keep monitoring regularly."
    }
}

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #b7f3c6, #8fd19e, #62b67a);
            background-attachment: fixed;
        }
        .glass-box {
            background: rgba(255, 255, 255, 0.85);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(8px);
        }
        .main-title {
            text-align: center;
            color: #166534;
            font-size: 40px;
            font-weight: bold;
        }
        .subtext {
            text-align: center;
            color: #374151;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .stProgress > div > div > div > div {
            background-color: #22c55e;
        }
        [data-testid="stSidebar"] {
            background-color: rgba(232, 255, 237, 0.9);
            color: #064e3b;
        }
        .info-card {
            background: rgba(240, 253, 244, 0.8);
            padding: 20px;
            border-left: 5px solid #16a34a;
            border-radius: 10px;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("🌱 About This Project")
st.sidebar.markdown("""
This web application identifies **plant leaf diseases** using a **MobileNetV2** deep learning model trained on the **PlantVillage Dataset**.

**Features:**
- 📸 Upload leaf images  
- 🔍 AI-powered disease detection  
- 📊 Confidence score display  
- 💊 Detailed disease info and treatments  
""")

st.sidebar.info("Model Accuracy: ~92.6%")
st.sidebar.caption("Powered by TensorFlow • Streamlit • Deep Learning")

# ---------------------- MAIN PAGE ----------------------
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Upload a leaf image to identify possible plant diseases instantly using AI.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)
    st.write("---")

    # Preprocess Image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🤖 Analyzing... Please wait."):
        pred = model.predict(img_array)
        result = np.argmax(pred)
        confidence = np.max(pred) * 100

    predicted_class = class_labels[result].strip()

    # Display Result
    st.success(f"🌱 **Predicted Disease:** {predicted_class}")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Display Information Card
    if predicted_class in disease_info:
        info = disease_info[predicted_class]
        st.markdown("---")
        st.subheader("🧬 Disease Information Card")
        st.markdown(f"""
        <div class='info-card'>
        <b>🧫 Cause:</b> {info['cause']}<br><br>
        <b>⚕️ Symptoms:</b> {info['symptoms']}<br><br>
        <b>💊 Treatment:</b> {info['treatment']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No detailed information available for this disease class.")

else:
    st.info("📥 Please upload a clear leaf image from your dataset to start detection.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("""
---
<div style='text-align: center; color: #065f46; font-size: 14px;'>
© 2025 | Deep Learning Based Plant Disease Detection 🌿
</div>
""", unsafe_allow_html=True)
