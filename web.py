import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

# Page configuration
st.set_page_config(
    page_title="TB Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-text {
        font-size: 18px;
        color: #2c3e50;
        font-weight: 500;
    }
    .header-title {
        font-size: 48px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .header-subtitle {
        font-size: 20px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 About This App")
    st.markdown("""
    This AI-powered application helps detect **Tuberculosis (TB)** from chest X-ray images using a Convolutional Neural Network (CNN).
    """)

    st.markdown("---")

    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload** a chest X-ray image (JPG, PNG, JPEG)
    2. **Wait** for the AI to analyze the image
    3. **Review** the prediction results
    4. **Consult** a healthcare professional for confirmation
    """)

    st.markdown("---")

    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    This tool is for **educational purposes** only. Always consult qualified healthcare professionals for medical diagnosis and treatment.
    """)

    st.markdown("---")

    st.markdown("### 👨‍💻 Developer")
    st.info("**Built by Jayes**")
    st.markdown("🔬 Powered by Deep Learning & CNN")

# Model loading with caching
@st.cache_resource
def loading_model():
    fp = "./model/model.h5"
    model_loader = load_model(fp)
    return model_loader

# Main header
st.markdown('<h1 class="header-title">🫁 TB Detection AI System</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Advanced Tuberculosis Detection Using Deep Learning</p>', unsafe_allow_html=True)

st.markdown("---")

# Load model with spinner
with st.spinner('🔄 Loading AI Model...'):
    cnn = loading_model()

st.success('✅ AI Model Loaded Successfully!')

# Create columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    st.markdown('<div class="info-box">Please upload a clear chest X-ray image for analysis. Supported formats: JPG, PNG, JPEG</div>', unsafe_allow_html=True)

    temp = st.file_uploader(
        "Choose an X-ray image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray image for TB detection"
    )

with col2:
    st.markdown("### ℹ️ Important Information")
    st.markdown("""
    <div class="warning-box">
    <strong>⚕️ Medical Advisory:</strong><br>
    • This is a screening tool, not a diagnostic device<br>
    • Always seek professional medical advice<br>
    • Results should be verified by qualified radiologists<br>
    • Early detection saves lives
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# File processing
buffer = temp
if buffer is not None:
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(buffer.getvalue())

    # Display uploaded image
    st.markdown("### 🖼️ Uploaded X-Ray Image")
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(image.load_img(temp_file.name), caption="Your Uploaded X-Ray", use_column_width=True)

    st.markdown("---")

    # Process the image
    with st.spinner('🔍 Analyzing X-Ray Image... Please wait...'):
        img = image.load_img(temp_file.name, target_size=(500, 500), color_mode='grayscale')

        # Preprocessing the image
        pp_img = image.img_to_array(img)
        pp_img = pp_img / 255
        pp_img = np.expand_dims(pp_img, axis=0)

        # Predict
        preds = cnn.predict(pp_img)
        confidence = float(preds[0][0])

    # Display results
    st.markdown("### 📊 Analysis Results")

    # Create columns for results
    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:
        if confidence >= 0.5:
            # TB Detected
            st.markdown(f"""
            <div class="danger-box">
            <h3 style="color: #dc3545; margin-top: 0;">⚠️ TB Case Detected</h3>
            <p style="font-size: 18px; margin-bottom: 10px;">
            The AI model is <strong>{confidence*100:.2f}%</strong> confident that this X-ray shows signs of Tuberculosis.
            </p>
            <p style="font-size: 16px; color: #721c24;">
            <strong>⚕️ Recommendation:</strong> Please consult a healthcare professional immediately for proper diagnosis and treatment.
            </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Normal Case
            normal_confidence = 1 - confidence
            st.markdown(f"""
            <div class="success-box">
            <h3 style="color: #28a745; margin-top: 0;">✅ Normal Case</h3>
            <p style="font-size: 18px; margin-bottom: 10px;">
            The AI model is <strong>{normal_confidence*100:.2f}%</strong> confident that this X-ray appears normal.
            </p>
            <p style="font-size: 16px; color: #155724;">
            <strong>ℹ️ Note:</strong> While the results look promising, regular health check-ups are still recommended.
            </p>
            </div>
            """, unsafe_allow_html=True)

    with result_col2:
        # Confidence meter
        st.markdown("#### 📈 Confidence Level")
        if confidence >= 0.5:
            st.progress(confidence)
            st.metric("TB Probability", f"{confidence*100:.1f}%", delta=None)
        else:
            st.progress(1 - confidence)
            st.metric("Normal Probability", f"{(1-confidence)*100:.1f}%", delta=None)

    st.markdown("---")

    # Additional statistics
    st.markdown("### 📈 Detailed Analysis")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("TB Probability", f"{confidence*100:.2f}%")

    with stat_col2:
        st.metric("Normal Probability", f"{(1-confidence)*100:.2f}%")

    with stat_col3:
        if confidence >= 0.5:
            st.metric("Diagnosis", "TB Detected", delta="High Risk", delta_color="inverse")
        else:
            st.metric("Diagnosis", "Normal", delta="Low Risk", delta_color="normal")

    with stat_col4:
        if confidence >= 0.5:
            st.metric("Confidence", "High" if confidence > 0.75 else "Moderate")
        else:
            st.metric("Confidence", "High" if (1-confidence) > 0.75 else "Moderate")

else:
    # Welcome screen when no image is uploaded
    st.markdown("### 👋 Welcome!")
    st.markdown("""
    <div class="info-box">
    <h4 style="margin-top: 0;">🎯 Get Started</h4>
    <p style="font-size: 16px;">
    This AI-powered application uses advanced deep learning to help detect Tuberculosis from chest X-ray images.
    Upload an X-ray image to begin the analysis.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Display sample instructions with icons
    st.markdown("### 🎓 Quick Guide")

    guide_col1, guide_col2, guide_col3 = st.columns(3)

    with guide_col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 48px;">📁</div>
        <h4>Step 1</h4>
        <p>Upload X-Ray Image</p>
        </div>
        """, unsafe_allow_html=True)

    with guide_col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 48px;">🤖</div>
        <h4>Step 2</h4>
        <p>AI Analyzes Image</p>
        </div>
        """, unsafe_allow_html=True)

    with guide_col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 48px;">📋</div>
        <h4>Step 3</h4>
        <p>Get Results</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 20px;">
<p style="font-size: 14px;">
🔬 Powered by TensorFlow & Keras | 🧠 Deep Learning CNN Model | 💻 Built by Jayes
</p>
<p style="font-size: 12px;">
⚠️ For educational and research purposes only. Not intended for clinical use.
</p>
</div>
""", unsafe_allow_html=True)
