import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import concurrent.futures
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI';
}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0b1f2a, #133b5c, #1f5f73);
    color: white;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: 800;
    color: #00ff9f;
    margin-top: -10px;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 22px;
    color: #dcdcdc;
    margin-bottom: 40px;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    margin-top: -10px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#07151f,#102c3d) !important;
    color: white !important;
}

/* Upload Area */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
}

/* Metrics */
.metric-box {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

/* Download Button */
.stDownloadButton > button {
    width: 100%;
    background: linear-gradient(90deg,#00c853,#64dd17);
    color: black !important;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 12px;
}

.stDownloadButton > button:hover {
    background: linear-gradient(90deg,#64dd17,#00c853);
    color: black !important;
}
            .card:hover {
    transform: scale(1.02);
    transition: 0.3s;
}

/* Table */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}
    section[data-testid="stSidebar"] * {
    color: white !important;
}  

            /* Sidebar Text Fix */

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Upload Box Text */

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: black !important;
    font-weight: bold;
}

/* Upload Icon */

[data-testid="stFileUploader"] svg {
    fill: black !important;
}

/* Upload Box */

[data-testid="stFileUploader"] section {
    background-color: white !important;
    border-radius: 15px;
    padding: 15px;
}    
            /* Upload Title White */

.stMarkdown h2 {
    color: white !important;
}

/* Hide Streamlit Header/Navbar */

header[data-testid="stHeader"] {
    display: none;
}

[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stDecoration"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}       

</style>
""", unsafe_allow_html=True)


# ================= LOAD MODEL =================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.applications import MobileNetV2

# ================= MODEL =================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
    InputLayer(shape=(224, 224, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(15, activation='softmax')
])

# ================= LOAD WEIGHTS =================

model.load_weights("model.weights.h5")# ================= CLASSES =================

classes = [
    "Pepper Bell Bacterial Spot",
    "Healthy Pepper Bell",

    "Potato Early Blight",
    "Potato Late Blight",
    "Healthy Potato",

    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy Tomato"
]

# ================= SIDEBAR =================

st.sidebar.markdown("""
# 🌿 Plant AI System

Advanced Deep Learning System for Plant Disease Detection
""")

st.sidebar.markdown("---")

st.sidebar.success("✅ AI Model Loaded")

st.sidebar.info("""
### 🚀 Technologies Used

✅ Computer Vision  
✅ Deep Learning CNN  
✅ Parallel Processing  
✅ Fuzzy Logic System  
✅ Streamlit Dashboard  
""")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📌 Features

- Multiple Image Upload
- Real-time Detection
- Severity Analysis
- Charts & Analytics
- CSV Report Download
""")

st.sidebar.markdown("---")

st.sidebar.warning("⚡ GPU Recommended For Faster Predictions")

st.sidebar.markdown("---")

st.sidebar.caption(f"🕒 {datetime.now().strftime('%d %B %Y')}")

# ================= TITLE =================

st.markdown(
    '<div class="main-title">🌿 Plant Disease Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Computer Vision + Parallel Distributed Computing + Fuzzy Logic</div>',
    unsafe_allow_html=True
)

# ================= FUZZY LOGIC =================

def fuzzy_severity(confidence):

    if confidence < 50:
        return "Low 🟢"

    elif confidence < 80:
        return "Medium 🟡"

    else:
        return "High 🔴"

# ================= PREDICTION FUNCTION =================



def predict_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")

    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    # ================= INVALID IMAGE CHECK =================

    if confidence < 60:

        disease = "❌ Not a Valid Leaf Image"

        severity = "None"

    else:

        # ================= SAFE CLASS CHECK =================

        if predicted_class >= len(classes):

            disease = "Unknown Disease"

        else:

            disease = classes[predicted_class]

        # ================= FUZZY LOGIC =================

        severity = fuzzy_severity(confidence)

    return {

        "image": image,

        "disease": disease,

        "confidence": round(confidence, 2),

        "severity": severity
    }
# ================= FILE UPLOADER =================


# ================= UPLOAD SECTION UI =================

st.markdown("""
<style>

/* Upload Section Container */

.upload-container {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 22px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 6px 25px rgba(0,0,0,0.35);
    backdrop-filter: blur(10px);
}

/* Upload Title */

.upload-title {
    color: #00ff99;
    font-size: 34px;
    font-weight: 800;
    text-align: center;
    text-shadow: 0px 0px 14px rgba(0,255,153,0.8);
    margin-bottom: 10px;
}

/* Upload Subtitle */

.upload-subtitle {
    color: #e0e0e0;
    text-align: center;
    font-size: 17px;
    margin-bottom: 10px;
}

/* File Uploader Box */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 18px !important;
    padding: 12px !important;
    border: none !important;
}

/* Upload Text */

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: black !important;
    font-weight: 700 !important;
}

/* Upload Icon */

[data-testid="stFileUploader"] svg {
    fill: black !important;
}

/* Remove Streamlit Header */

header[data-testid="stHeader"] {
    display: none;
}

[data-testid="stToolbar"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ================= CUSTOM UPLOAD HEADING =================

st.markdown("""
<div class="upload-container">

<div class="upload-title">
📂 Upload Plant Leaf Images
</div>

<div class="upload-subtitle">
Upload multiple plant leaf images for AI disease detection and severity analysis
</div>

</div>
""", unsafe_allow_html=True)

# ================= FILE UPLOADER =================

uploaded_files = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ================= MAIN =================

if uploaded_files:

    st.success(f"✅ {len(uploaded_files)} Images Uploaded Successfully")

    progress_bar = st.progress(0)

    results = []

    # ================= PARALLEL PROCESSING =================

    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = [
            executor.submit(predict_image, file)
            for file in uploaded_files
        ]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):

            results.append(future.result())

            progress_bar.progress((i + 1) / len(uploaded_files))

    st.markdown("---")

    # ================= METRICS =================

    total_images = len(results)

    avg_confidence = np.mean([r["confidence"] for r in results])

    high_cases = sum(
        1 for r in results if "High" in r["severity"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
        <h2>📸</h2>
        <h1>{total_images}</h1>
        <p>Total Images</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
        <h2>🎯</h2>
        <h1>{avg_confidence:.2f}%</h1>
        <p>Average Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
        <h2>⚠️</h2>
        <h1>{high_cases}</h1>
        <p>High Severity Cases</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ================= RESULT SECTION =================

    st.markdown("## 🌱 Detection Results")

    cols = st.columns(3)

    for idx, result in enumerate(results):

        with cols[idx % 3]:

            st.image(
                result["image"],
                use_container_width=True
            )

            severity_color = {
                "Low 🟢": "#00e676",
                "Medium 🟡": "#ffea00",
                "High 🔴": "#ff1744"
            }

            color = severity_color.get(
                result["severity"],
                "white"
            )

            st.markdown(f"""
            <div class="card">

            <h2 style="color:#00ff9f;">
            🌿 {result['disease']}
            </h2>

            <hr>

            <h3>🎯 Confidence</h3>

            <p style="
                font-size:28px;
                color:#FFD700;
                font-weight:bold;
            ">
            {result['confidence']}%
            </p>

            <h3>⚠️ Severity</h3>

            <p style="
                font-size:24px;
                font-weight:bold;
                color:{color};
            ">
            {result['severity']}
            </p>

            </div>
            """, unsafe_allow_html=True)

    # ================= TABLE =================

    st.markdown("## 📊 Detection Summary Table")

    table_data = []

    for r in results:

        table_data.append({
            "Disease": r["disease"],
            "Confidence (%)": r["confidence"],
            "Severity": r["severity"]
        })

    df = pd.DataFrame(table_data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # ================= CHARTS =================

    st.markdown("## 📈 Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        disease_count = df["Disease"].value_counts().reset_index()

        disease_count.columns = ["Disease", "Count"]

        fig1 = px.bar(
            disease_count,
            x="Disease",
            y="Count",
            title="Disease Distribution",
            text_auto=True
        )

        fig1.update_layout(
    title_font_color="white",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
      )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        fig2 = px.pie(
            df,
            names="Severity",
            title="Severity Analysis",
            hole=0.4
        )

        fig2.update_layout(
    title_font_color="white",
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # ================= EXTRA CHART =================

    st.markdown("## 📉 Confidence Analysis")

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        y=df["Confidence (%)"],
        mode='lines+markers',
        name='Confidence'
    ))

    fig3.update_layout(
    title={
        'text': "Prediction Confidence Trend",
        'font': {'color': 'white', 'size': 24}
    },
    xaxis_title="Images",
    yaxis_title="Confidence %",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

    st.plotly_chart(fig3, use_container_width=True)

    # ================= DOWNLOAD =================

    st.markdown("## 📥 Download Report")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Detection Report CSV",
        data=csv,
        file_name="plant_disease_report.csv",
        mime="text/csv"
    )

    st.success("✅ All Images Processed Successfully!")

# ================= FOOTER =================

st.markdown("---")

st.markdown("""
<div style='text-align:center;'>

<h3>🌿 AI Powered Plant Disease Detection System</h3>

<p>
Computer Vision | Parallel Processing | Fuzzy Logic
</p>

<p style='color:gray;'>
Developed Using TensorFlow + Streamlit
</p>

</div>
""", unsafe_allow_html=True)