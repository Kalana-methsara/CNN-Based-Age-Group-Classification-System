import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import gdown
import os
import pandas as pd

GOOGLE_DRIVE_FILE_ID = "1reGBQyBks1mIMy05l5J7qUD0dslvN020"
MODEL_PATH = "checkpoint.pt"

AGE_GROUPS = [
    "0–2", "3–9", "10–19", "20–29",
    "30–39", "40–49", "50–59", "60–69", "70+"
]

st.set_page_config(
    page_title="FairVision AI - Anti-Light Mode Premium Theme",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
    /* 1. FORCE CORE LAYOUT TO DARK (Overriding global structural variables) */
    html, body, [data-testid="stAppViewContainer"], .main {
        --background-color: #0B0F19 !important;
        --secondary-background-color: #111827 !important;
        --text-color: #E5E7EB !important;
        --primary-color: #00F2FE !important;
        background-color: #0B0F19 !important;
        color: #E5E7EB !important;
    }
    
    /* 2. THE TOP HEADER FIX (Eliminates the light mode white bar at the top) */
    header[data-testid="stHeader"] {
        background-color: #0B0F19 !important;
        background: #0B0F19 !important;
    }
    
    /* 3. FIXED SIDEBAR (Forces dark layout under any condition) */
    [data-testid="stSidebar"], [data-testid="stSidebarUserContent"], section[data-testid="stSidebar"] > div {
        background-color: #111827 !important;
        border-right: 2px solid #1F2937 !important;
    }
    
    /* 4. THE FILE UPLOADER FIX (Fixes the white file box in Light Mode) */
    [data-testid="stFileUploader"] section {
        background-color: #1F2937 !important;
        border: 1px dashed rgba(0, 242, 254, 0.4) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #E5E7EB !important;
    }
    /* Upload button inside the uploader */
    [data-testid="stFileUploader"] button {
        background-color: #111827 !important;
        color: #00F2FE !important;
        border: 1px solid #1F2937 !important;
    }
    
    /* Global Text Elements Overrides */
    h1, h2, h3, h4, h5, h6, p, span, label, li, small {
        color: #E5E7EB !important;
    }
    
    /* Main Gradient Title */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-top: 10px;
    }
    
    .subtitle {
        font-size: 1.1rem !important;
        color: #9CA3AF !important;
        margin-bottom: 2.5rem;
    }
    
    /* Premium Neon Glass Cards */
    .neon-card {
        background: rgba(17, 24, 39, 0.9) !important;
        border: 1px solid rgba(79, 172, 254, 0.4) !important;
        border-radius: 16px !important;
        padding: 22px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.08) !important;
        margin-bottom: 15px;
    }
    
    .neon-card-alt {
        background: rgba(17, 24, 39, 0.9) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 16px !important;
        padding: 22px !important;
        box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.08) !important;
        margin-bottom: 15px;
    }
    
    .card-label {
        font-size: 0.85rem !important;
        color: #9CA3AF !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-value {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin-top: 8px;
    }
    
    /* Headers with Icons styling */
    .custom-header {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #F3F4F6 !important;
        border-bottom: 2px solid #1F2937 !important;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .img-box {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid #1F2937;
        background-color: #111827;
    }
    
    /* Sidebar text controls label fix */
    div[data-testid="stWidgetLabel"] p {
        color: #00F2FE !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

class FairVisionResNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = FairVisionResNet(num_classes=9)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state_dict.items()},
        strict=False
    )
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    return predicted.item(), confidence.item(), probabilities

with st.sidebar:
    st.markdown("<h2 style='color: #00F2FE;'><i class='fa-solid fa-sliders'></i> CONTROL PANEL</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9CA3AF;'>Upload an image to compute demographics analytics.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a face image:",
        type=["jpg", "jpeg", "png"]
    )
    
    st.markdown("<br><hr style='border-color: #1F2937;'><br>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #4FACFE;'><i class='fa-solid fa-circle-info'></i> About</h4>", unsafe_allow_html=True)
    st.caption("FairVision runs on a fine-tuned ResNet-50 deep learning model architecture.")

st.markdown('<div class="main-title"><i class="fa-solid fa-eye-low-vision"></i> FairVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Age Group Classification Framework</div>', unsafe_allow_html=True)

if uploaded_file:
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown('<div class="custom-header"><i class="fa-solid fa-camera"></i> Analyzed Image</div>', unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="img-box">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        with st.spinner("🧠 Running Deep Learning Inference..."):
            pred_idx, confidence, probs = predict(image)
        predicted_label = AGE_GROUPS[pred_idx]
        
        st.markdown('<div class="custom-header"><i class="fa-solid fa-chart-simple"></i> Classification Results</div>', unsafe_allow_html=True)
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown(f"""
                <div class="neon-card">
                    <div class="card-label"><i class="fa-solid fa-user-clock" style="color:#00F2FE;"></i> Predicted Age Group</div>
                    <div class="card-value" style="color: #00F2FE;">{predicted_label} Yrs</div>
                </div>
            """, unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown(f"""
                <div class="neon-card-alt">
                    <div class="card-label"><i class="fa-solid fa-shield-halved" style="color:#10B981;"></i> Confidence Score</div>
                    <div class="card-value" style="color: #10B981;">{confidence * 100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<div class="custom-header"><i class="fa-solid fa-bullseye"></i> Top 3 Probabilities</div>', unsafe_allow_html=True)
        top_probs, top_indices = torch.topk(probs, 3)
        
        for i in range(3):
            label = AGE_GROUPS[top_indices[i].item()]
            score = top_probs[i].item()
            
            p_col1, p_col2 = st.columns([3, 1])
            p_col1.markdown(f"**<i class='fa-solid fa-arrow-trend-up' style='color:#4FACFE; font-size:0.9rem;'></i> Age Group: {label}**", unsafe_allow_html=True)
            p_col2.write(f"`{score * 100:.2f}%`")
            st.progress(score)
            
        st.markdown('<div class="custom-header"><i class="fa-solid fa-chart-line"></i> Full Distribution</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Age Group": AGE_GROUPS,
            "Probability (%)": [p.item() * 100 for p in probs]
        })
        st.bar_chart(df.set_index("Age Group"), color="#4FACFE")

else:
    st.markdown("""
        <div style="background: #111827; border: 2px dashed #1F2937; padding: 40px; border-radius: 16px; text-align: center;">
            <h4 style="color: #9CA3AF; margin-bottom: 10px;"><i class="fa-solid fa-image" style="font-size:2rem; color:#4FACFE; margin-bottom:10px;"></i><br>💡 No Image Detected</h4>
            <p style="color: #6B7280; max-width:500px; margin:0 auto;">Please upload a portrait image file from the sidebar panel control center to initiate computer vision classifications.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-header"><i class="fa-solid fa-microchip"></i> System Specifications</div>', unsafe_allow_html=True)
    spec_col1, spec_col2, spec_col3 = st.columns(3)
    
    spec_col1.markdown('<div class="neon-card"><div class="card-label"><i class="fa-solid fa-brain"></i> Core Engine</div><div class="card-value" style="font-size:1.5rem;">ResNet-50</div></div>', unsafe_allow_html=True)
    spec_col2.markdown('<div class="neon-card"><div class="card-label"><i class="fa-solid fa-layer-group"></i> Target Classes</div><div class="card-value" style="font-size:1.5rem;">9 Groups</div></div>', unsafe_allow_html=True)
    spec_col3.markdown('<div class="neon-card"><div class="card-label"><i class="fa-solid fa-code"></i> Framework</div><div class="card-value" style="font-size:1.5rem;">PyTorch</div></div>', unsafe_allow_html=True)