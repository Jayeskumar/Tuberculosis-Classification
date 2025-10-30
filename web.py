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
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .result-box {
        padding: 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .normal-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("## 🏥 About This App")
    st.info("""
    This AI-powered application uses Convolutional Neural Networks (CNN) to detect
    Tuberculosis from chest X-ray images with high accuracy.
    """)

    st.markdown("## 📋 How to Use")
    st.markdown("""
    1. **Upload** a chest X-ray image
    2. **Wait** for AI analysis
    3. **Review** the results
    4. **Consult** a doctor for confirmation
    """)

    st.markdown("## ⚠️ Important Notice")
    st.warning("""
    This tool is for screening purposes only.
    Always consult with qualified healthcare
    professionals for proper diagnosis and treatment.
    """)

    st.markdown("## 👨‍💻 Developer")
    st.markdown("**Built by:** Jayes")
    st.markdown("**Version:** 2.0")

# Cache model loading
@st.cache_resource
def loading_model():
    fp = "./model/model.h5"
    model_loader = load_model(fp)
    return model_loader

# Header
st.markdown("""
    <div class="header-container">
        <h1>🫁 Tuberculosis Detection AI System</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Cloud-Based Deep Learning Solution for TB Screening
        </p>
    </div>
""", unsafe_allow_html=True)

# Load model with progress
with st.spinner("🔄 Loading AI model..."):
    cnn = loading_model()

st.success("✅ Model loaded successfully!")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    st.markdown("""
        <div class="info-card">
            <strong>Supported formats:</strong> JPG, JPEG, PNG<br>
            <strong>Image type:</strong> Chest X-Ray (Frontal view preferred)<br>
            <strong>Quality:</strong> Clear, well-lit images work best
        </div>
    """, unsafe_allow_html=True)

    temp = st.file_uploader(
        "Choose an X-ray image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray image for TB detection"
    )

with col2:
    st.markdown("### 📊 Quick Stats")
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.markdown("""
            <div class="metric-container">
                <h2 style="color: #667eea; margin: 0;">CNN</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">AI Model</p>
            </div>
        """, unsafe_allow_html=True)

    with metric_col2:
        st.markdown("""
            <div class="metric-container">
                <h2 style="color: #667eea; margin: 0;">500x500</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Resolution</p>
            </div>
        """, unsafe_allow_html=True)

    with metric_col3:
        st.markdown("""
            <div class="metric-container">
                <h2 style="color: #667eea; margin: 0;">Fast</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Analysis</p>
            </div>
        """, unsafe_allow_html=True)

# Processing section
if temp is None:
    st.markdown("---")
    st.info("👆 Please upload an X-ray image to begin analysis")

    # Show sample information
    st.markdown("### 🔍 What We Analyze")
    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown("""
            <div class="info-card">
                <h4>🎯 Pattern Recognition</h4>
                <p>Advanced CNN identifies TB-specific patterns in lung tissue</p>
            </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
            <div class="info-card">
                <h4>📈 Confidence Scoring</h4>
                <p>Provides percentage-based confidence levels for predictions</p>
            </div>
        """, unsafe_allow_html=True)

    with info_col3:
        st.markdown("""
            <div class="info-card">
                <h4>⚡ Quick Results</h4>
                <p>Get instant analysis results within seconds</p>
            </div>
        """, unsafe_allow_html=True)

else:
    # Process the uploaded image
    st.markdown("---")
    st.markdown("### 🔬 Analysis in Progress")

    # Create temporary file
    buffer = temp
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(buffer.getvalue())

    # Display uploaded image
    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        st.markdown("#### 📸 Uploaded X-Ray Image")
        img_display = Image.open(temp)
        st.image(img_display, use_column_width=True, caption="Original X-Ray Image")

    with result_col2:
        st.markdown("#### 🤖 AI Analysis")

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate analysis steps with progress
        status_text.text("Loading image...")
        progress_bar.progress(25)

        # Load and preprocess image
        img = image.load_img(temp_file.name, target_size=(500, 500), color_mode='grayscale')

        status_text.text("Preprocessing image...")
        progress_bar.progress(50)

        # Preprocessing the image
        pp_img = image.img_to_array(img)
        pp_img = pp_img/255
        pp_img = np.expand_dims(pp_img, axis=0)

        status_text.text("Running AI model...")
        progress_bar.progress(75)

        # Make prediction
        preds = cnn.predict(pp_img, verbose=0)

        status_text.text("Analysis complete!")
        progress_bar.progress(100)

        # Show preprocessed image
        st.image(img, use_column_width=True, caption="Processed Image (Grayscale)")

    # Display results
    st.markdown("---")
    st.markdown("### 📋 Diagnosis Results")

    # Determine result
    confidence = float(preds[0][0])
    is_tb = confidence >= 0.5

    if is_tb:
        result_confidence = confidence * 100
        st.markdown(f"""
            <div class="result-box">
                ⚠️ TUBERCULOSIS DETECTED<br>
                <span style="font-size: 2rem; margin: 1rem 0; display: block;">{result_confidence:.1f}%</span>
                Confidence Level
            </div>
        """, unsafe_allow_html=True)

        st.error(f"""
        **Analysis Result:** The AI model has detected patterns consistent with Tuberculosis.

        **Confidence Level:** {result_confidence:.2f}%

        **Recommendation:** Please consult a healthcare professional immediately for proper diagnosis and treatment.
        """)

        # Show confidence meter
        st.markdown("#### 📊 Detailed Confidence Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("TB Probability", f"{result_confidence:.2f}%", delta="High Risk", delta_color="inverse")
        with col_b:
            st.metric("Normal Probability", f"{(100-result_confidence):.2f}%", delta="Low", delta_color="normal")

    else:
        result_confidence = (1 - confidence) * 100
        st.markdown(f"""
            <div class="result-box normal-box">
                ✅ NORMAL RESULT<br>
                <span style="font-size: 2rem; margin: 1rem 0; display: block;">{result_confidence:.1f}%</span>
                Confidence Level
            </div>
        """, unsafe_allow_html=True)

        st.success(f"""
        **Analysis Result:** The AI model indicates this appears to be a normal chest X-ray.

        **Confidence Level:** {result_confidence:.2f}%

        **Note:** While the result is encouraging, regular health check-ups are always recommended.
        """)

        # Show confidence meter
        st.markdown("#### 📊 Detailed Confidence Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Normal Probability", f"{result_confidence:.2f}%", delta="Low Risk", delta_color="normal")
        with col_b:
            st.metric("TB Probability", f"{(100-result_confidence):.2f}%", delta="Low", delta_color="inverse")

    # Show confidence chart
    st.markdown("#### 📈 Confidence Visualization")
    chart_data = {
        'Category': ['Tuberculosis', 'Normal'],
        'Probability': [confidence * 100, (1 - confidence) * 100]
    }
    st.bar_chart(chart_data, x='Category', y='Probability', use_container_width=True)

    # Next steps
    st.markdown("---")
    st.markdown("### 🎯 Recommended Next Steps")

    if is_tb:
        st.markdown("""
        1. **🏥 Consult a Doctor:** Schedule an appointment with a pulmonologist or TB specialist
        2. **🔬 Additional Tests:** Get sputum tests, CT scans, or other confirmatory diagnostics
        3. **💊 Treatment Plan:** Follow prescribed treatment regimen if diagnosed
        4. **👥 Contact Tracing:** Inform close contacts to get screened
        5. **📅 Follow-up:** Regular monitoring and follow-up appointments
        """)
    else:
        st.markdown("""
        1. **✅ Stay Healthy:** Continue maintaining good health practices
        2. **🔄 Regular Screening:** Get routine check-ups as recommended
        3. **😷 Preventive Care:** Practice good hygiene and avoid high-risk exposures
        4. **🏃 Healthy Lifestyle:** Maintain a balanced diet and regular exercise
        5. **📞 Any Concerns:** Consult a doctor if you experience any symptoms
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Disclaimer:</strong> This AI system is designed to assist in TB screening but should not replace professional medical diagnosis.</p>
        <p>Always consult qualified healthcare professionals for accurate diagnosis and treatment.</p>
        <p style="margin-top: 1rem;">© 2024 TB Detection AI | Built by Jayes | Powered by Deep Learning</p>
    </div>
""", unsafe_allow_html=True)
