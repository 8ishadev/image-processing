import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Image Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .filter-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .download-btn {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
    }
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .small-image {
        max-width: 300px !important;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient text
st.markdown('<h1 class="main-header">✨ Image Studio</h1>', unsafe_allow_html=True)
st.markdown("### Transform your images with powerful filters in real-time! 🎨")

# Sidebar for navigation and filters
with st.sidebar:
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Control Panel")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File uploader in sidebar
    uploaded_file = st.file_uploader("📁 Upload Your Image", 
                                   type=['jpg', 'jpeg', 'png'],
                                   help="Supported formats: JPG, JPEG, PNG")
    
    st.markdown("---")
    
    # Filter options
    st.markdown("### 🔮Filters")
    filter_choice = st.selectbox(
        "Choose your filter :",
        ["None", "Grayscale", "Blur", "Edge Detection", "Sharpen", 
         "Sepia", "Invert Colors", "Sketch Effect", "Emboss"],
        help="Select a filter to apply to your image"
    )
    
    # Filter intensity controls
    intensity = 3
    if filter_choice in ["Blur"]:
        intensity = st.slider("🎚️ Blur Intensity", 1, 10, 3, 
                            help="Adjust the blur strength")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("• Upload any JPG/PNG image\n• Compare original vs filtered\n• Download your masterpiece!")

def resize_image(image, max_width=300):  # CHANGED TO 300 FOR SMALLER IMAGES
    """Resize image to manageable size while maintaining aspect ratio"""
    if image.shape[1] > max_width:
        ratio = max_width / image.shape[1]
        new_width = max_width
        new_height = int(image.shape[0] * ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def apply_grayscale(image):
    """Convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(image, intensity=3):
    """Apply Gaussian blur"""
    # Ensure kernel size is odd
    kernel_size = intensity * 2 + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_edge_detection(image):
    """Apply Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_sharpen(image):
    """Apply sharpening filter"""
    kernel = np.array([[-1,-1,-1], 
                       [-1,9,-1], 
                       [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def apply_sepia(image):
    """Apply sepia filter"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, kernel)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

def apply_invert(image):
    """Invert colors"""
    return cv2.bitwise_not(image)

def apply_sketch(image):
    """Apply pencil sketch effect"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_emboss(image):
    """Apply emboss filter"""
    kernel = np.array([[-2,-1,0], 
                       [-1,1,1], 
                       [0,1,2]])
    return cv2.filter2D(image, -1, kernel)

# Main content area
if uploaded_file is not None:
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to numpy array and handle different formats
        opencv_image = np.array(image)
        
        # Handle different image formats
        if len(opencv_image.shape) == 3:
            if opencv_image.shape[2] == 4:  # RGBA image
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGR)
            else:  # RGB image
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        else:  # Grayscale image
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)
        
        # Resize image to manageable size - CHANGED TO SMALLER SIZE
        opencv_image = resize_image(opencv_image, max_width=300)  # Smaller size
        
        # Apply selected filter
        processed_image = opencv_image.copy()
        
        if filter_choice == "Grayscale":
            processed_image = apply_grayscale(processed_image)
            if len(processed_image.shape) == 2:  # If grayscale
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        
        elif filter_choice == "Blur":
            processed_image = apply_blur(processed_image, intensity)
        
        elif filter_choice == "Edge Detection":
            processed_image = apply_edge_detection(processed_image)
        
        elif filter_choice == "Sharpen":
            processed_image = apply_sharpen(processed_image)
        
        elif filter_choice == "Sepia":
            processed_image = apply_sepia(processed_image)
        
        elif filter_choice == "Invert Colors":
            processed_image = apply_invert(processed_image)
        
        elif filter_choice == "Sketch Effect":
            processed_image = apply_sketch(processed_image)
        
        elif filter_choice == "Emboss":
            processed_image = apply_emboss(processed_image)
        
        # Display images in columns with smaller size
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="sub-header">📷 Original Image</div>', unsafe_allow_html=True)
            display_original = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            # Use width parameter to control image size
            st.image(display_original, width=300, caption="Your original masterpiece")
        
        with col2:
            st.markdown(f'<div class="sub-header">🎨 {filter_choice} Filter</div>', unsafe_allow_html=True)
            display_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            # Use width parameter to control image size
            st.image(display_processed, width=300, caption=f"With {filter_choice} applied")
        
        # Image information cards
        st.markdown("---")
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        
        with col_info1:
            st.metric("📐 Display Size", f"{opencv_image.shape[1]}x{opencv_image.shape[0]}")
        
        with col_info2:
            st.metric("📊 File Type", uploaded_file.type.split('/')[-1].upper())
        
        with col_info3:
            st.metric("🔮 Filter Applied", filter_choice if filter_choice != "None" else "Original")
        
        with col_info4:
            file_size = len(uploaded_file.getvalue()) // 1024
            st.metric("💾 File Size", f"{file_size} KB")
        
        # Download section
        st.markdown("---")
        st.markdown('<div class="sub-header">📥 Download Your Creation</div>', unsafe_allow_html=True)
        
        # Convert processed image to PIL format for download
        download_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(download_image)
        
        # Convert to bytes
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=95)
        byte_im = buf.getvalue()
        
        col_dl1, col_dl2, col_dl3 = st.columns([1,2,1])
        with col_dl2:
            st.download_button(
                label="🚀 Download Processed Image",
                data=byte_im,
                file_name=f"magic_{uploaded_file.name}",
                mime="image/jpeg",
                use_container_width=True,
                key="download_btn"
            )
        
    except Exception as e:
        st.error(f"❌ Oops! Something went wrong: {str(e)}")
        st.info("💡 Please try uploading a different image file or check the file format.")

else:
    # Welcome section when no image is uploaded
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;'>
        <h2>🎨 Ready to Create Magic?</h2>
        <p style='font-size: 1.2rem; margin-bottom: 2rem;'>Upload an image and watch the transformation begin!</p>
        <div style='font-size: 3rem; margin: 2rem 0;'>
            ✨ 🖼️ 🎭 🔮 🌟
        </div>
        <p style='font-size: 1.1rem;'>Select from our collection of amazing filters and transform your ordinary photos into extraordinary artworks!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    st.markdown("### 🚀 Why Choose Image Studio?")
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;'>
            <h3>⚡ Real-time Preview</h3>
            <p>See changes instantly as you apply different filters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_feat2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;'>
            <h3>🎭 8+ Amazing Filters</h3>
            <p>From classic B&W to artistic sketches</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_feat3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;'>
            <h3>💾 Easy Download</h3>
            <p>Save your creations with one click</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem;'>"
    "Made with ❤️ using Streamlit | Transform your images! ✨"
    "</div>", 
    unsafe_allow_html=True
)