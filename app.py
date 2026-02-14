import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(BASE_DIR,"model.h5"))

class_indices = json.load(open(os.path.join(BASE_DIR,"class_indices.json")))
class_indices = {int(k):v for k,v in class_indices.items()}

# REMEDIES DICTIONARY 

remedies = {
    'Apple___Apple_scab': 'Remedy: Rake up and destroy infected leaves. Prune trees to improve air circulation. Apply fungicides like captan or sulfur.',
    'Apple___Black_rot': 'Remedy: Prune out dead or diseased branches. Remove and destroy mummified fruits. Apply appropriate fungicides during the growing season.',
    'Apple___Cedar_apple_rust': 'Remedy: Remove nearby cedar trees if possible. Apply fungicides from pink bud stage until petals fall. Use rust-resistant apple varieties.',
    'Apple___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Blueberry___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Cherry_(including_sour)___Powdery_mildew': 'Remedy: Ensure good air circulation by pruning. Apply fungicides like sulfur, potassium bicarbonate, or neem oil.',
    'Cherry_(including_sour)___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Remedy: Practice crop rotation. Use resistant hybrids. Apply fungicides when disease first appears.',
    'Corn_(maize)___Common_rust_': 'Remedy: Plant resistant hybrids. Fungicide applications are often not economically necessary but can be used in severe cases.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Remedy: Use resistant corn varieties. Practice crop rotation and tillage to bury crop residue. Apply fungicides if necessary.',
    'Corn_(maize)___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Grape___Black_rot': 'Remedy: Prune vines in winter to remove cankers. Destroy mummified berries. Apply fungicides during the growing season.',
    'Grape___Esca_(Black_Measles)': 'Remedy: Prune out infected wood well beyond the visible symptoms. There is no chemical cure.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Remedy: Apply fungicides. Improve air circulation through canopy management.',
    'Grape___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Orange___Haunglongbing_(Citrus_greening)': 'Remedy: There is no cure for this disease. Remove and destroy infected trees to prevent spread. Control the Asian citrus psyllid insect vector.',
    'Peach___Bacterial_spot': 'Remedy: Use resistant varieties. Apply copper-based sprays. Avoid nitrogen-heavy fertilizers.',
    'Peach___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Pepper,_bell___Bacterial_spot': 'Remedy: Use disease-free seed and transplants. Avoid overhead irrigation. Apply copper-based bactericides.',
    'Pepper,_bell___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Potato___Early_blight': 'Remedy: Use certified disease-free seed potatoes. Practice crop rotation. Apply fungicides like mancozeb or chlorothalonil.',
    'Potato___Late_blight': 'Remedy: Destroy volunteer potato plants. Apply fungicides proactively, especially in cool, moist conditions.',
    'Potato___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Raspberry___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Soybean___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Squash___Powdery_mildew': 'Remedy: Ensure good air circulation. Water the soil, not the leaves. Apply fungicides like sulfur, neem oil, or potassium bicarbonate.',
    'Strawberry___Leaf_scorch': 'Remedy: Remove infected leaves. Ensure good air circulation. Use resistant varieties.',
    'Strawberry___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!',
    'Tomato___Bacterial_spot': 'Remedy: Use clean seed. Avoid working with plants when they are wet. Spray with copper-based bactericides.',
    'Tomato___Early_blight': 'Remedy: Mulch plants to reduce soil splash. Prune lower leaves. Ensure good air circulation. Apply fungicides.',
    'Tomato___Late_blight': 'Remedy: Plant resistant varieties. Ensure good spacing for air flow. Apply fungicides preventatively.',
    'Tomato___Leaf_Mold': 'Remedy: Provide good ventilation, especially in greenhouses. Stake plants. Apply fungicides.',
    'Tomato___Septoria_leaf_spot': 'Remedy: Remove infected leaves. Mulch around the base of plants. Apply fungicides.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Remedy: Spray plants with a strong stream of water. Use insecticidal soaps or neem oil. Encourage natural predators.',
    'Tomato___Target_Spot': 'Remedy: Improve air circulation. Avoid overhead watering. Apply fungicides.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Remedy: Control whitefly populations, the vector of the virus. Remove and destroy infected plants.',
    'Tomato___Tomato_mosaic_virus': 'Remedy: There is no cure. Remove and destroy infected plants. Wash hands and tools frequently.',
    'Tomato___healthy': 'This plant is healthy. No remedy is needed. Keep up the good care!'
}

def get_remedy(disease):
    return remedies.get(disease,"Remedy information not available.")

def preprocess_image(image):
    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)
    return img_array

st.markdown("""
<style>

.stApp{
background:
radial-gradient(circle at 10% 20%,rgba(16,185,129,0.15),transparent 40%),
radial-gradient(circle at 90% 80%,rgba(34,197,94,0.12),transparent 40%),
linear-gradient(120deg,#f0fdf4,#ecfdf5);
}

.title{
font-size:44px;
font-weight:800;
text-align:center;
color:#065f46;
}

.subtitle{
text-align:center;
color:#047857;
margin-bottom:30px;
}

.card{
background:rgba(255,255,255,0.75);
backdrop-filter:blur(14px);
padding:28px;
border-radius:18px;
box-shadow:0 10px 30px rgba(0,0,0,0.08);
}

.good{
background:#dcfce7;
padding:22px;
border-radius:14px;
font-size:24px;
text-align:center;
color:#166534;
font-weight:700;
}

.bad{
background:#fee2e2;
padding:22px;
border-radius:14px;
font-size:24px;
text-align:center;
color:#991b1b;
font-weight:700;
}

</style>
""",unsafe_allow_html=True)

st.markdown('<div class="title">🌿 AI Plant Disease Detection</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning CNN trained on PlantVillage Dataset</div>',unsafe_allow_html=True)

col1,col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("📤 Upload Leaf Image")

    uploaded_file = st.file_uploader(
        "Upload plant leaf image",
        type=["jpg","jpeg","png"]
    )
    st.markdown('</div>',unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("🤖 Diagnosis Result")

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image,use_column_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)

        confidence = float(np.max(prediction))
        predicted_class = class_indices[np.argmax(prediction)]

        st.progress(confidence)

        if "healthy" in predicted_class.lower():
            st.markdown(
                f'<div class="good">✅ Healthy Plant — Confidence: {confidence:.2f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bad">⚠️ Disease Detected: {predicted_class}<br>Confidence: {confidence:.2f}</div>',
                unsafe_allow_html=True
            )

        st.write("### 🌱 Recommended Remedy")
        st.info(get_remedy(predicted_class))

    else:
        st.info("Upload an image to begin diagnosis.")

    st.markdown('</div>',unsafe_allow_html=True)
