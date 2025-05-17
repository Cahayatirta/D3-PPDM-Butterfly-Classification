import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import os
import io

# Import the ButterflyNN class from the existing module
from butterfly_nn_hyperparametertuning import ButterflyNN

class ButterflyClassifierApp:
    def __init__(self):
        # Load model
        self.model = None
        self.mean = None
        self.std = None
        self.class_names = None
        self.load_model()
    
    def load_model(self):
        """Load model from pickle file"""
        try:
            with open("butterfly_model_hyperparameter.pkl", 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.mean = data['mean']
                self.std = data['std']
                self.class_names = data['class_names']
                
                # Verify dimensions
                expected_dim = self.model.weights[0].shape[0]
                st.sidebar.success(f"Model loaded. Expected feature dimension: {expected_dim}")
                
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            
    def extract_hybrid_features(self, img_array):
        """Extract features from image"""
        hog_features = []
        color_features = []
        glcm_features = []
        
        for img in img_array:
            # 1. HOG Features
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_uint8 = (gray * 255).astype(np.uint8)
            hog = cv2.HOGDescriptor((96,96), (16,16), (8,8), (8,8), 9)
            hog_feat = hog.compute(gray_uint8).flatten()
            
            # 2. Color Histogram (HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hist = []
            for channel in range(3):
                hist_ch = np.histogram(hsv[:,:,channel], bins=16, range=(0,256))[0]
                hist.extend(hist_ch / (np.sum(hist_ch) + 1e-6))
            
            # 3. GLCM Features
            glcm = self.calculate_glcm(gray)
            
            hog_features.append(hog_feat)
            color_features.append(hist)
            glcm_features.append(glcm)
        
        return np.hstack([np.array(hog_features), 
                         np.array(color_features), 
                         np.array(glcm_features)])
    
    def calculate_glcm(self, image, levels=8):
        """Calculate GLCM features"""
        glcm = np.zeros((levels, levels), dtype=np.float32)
        image = (image * (levels-1)).astype(np.uint8)
        
        for i in range(image.shape[0]-1):
            for j in range(image.shape[1]-1):
                val1 = image[i,j]
                val2 = image[i+1,j]
                glcm[val1, val2] += 1
                
        glcm /= np.sum(glcm)
        
        # Calculate GLCM features
        contrast = np.sum((np.arange(levels)[:,None] - np.arange(levels))**2 * glcm)
        energy = np.sum(glcm**2)
        homogeneity = np.sum(glcm / (1 + (np.arange(levels)[:,None] - np.arange(levels))**2))
        
        return np.array([contrast, energy, homogeneity])
    
    def preprocess(self, img_array):
        """Normalize features"""
        features = self.extract_hybrid_features(img_array)
        return (features - self.mean) / (self.std + 1e-6)
    
    def predict(self, img):
        """Predict butterfly species"""
        # Convert to RGB if needed (PIL images are in RGB already)
        if isinstance(img, np.ndarray) and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        img = cv2.resize(img, (96, 96)).astype(np.float32) / 255.0
        
        # Extract features
        features = self.preprocess(np.array([img]))
        
        # Predict
        probs = self.model.forward(features, training=False)
        pred_class = np.argmax(probs)
        
        return self.class_names[pred_class], np.max(probs)

def main():
    st.set_page_config(
        page_title="Butterfly Species Classifier",
        page_icon="🦋",
        layout="wide"
    )
    
    # Create title and header with styling
    st.title("🦋 Butterfly Species Classifier")
    st.markdown("""
    <style>
    .main-header {
        color: #4a6baf;
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #4a6baf;
        font-size: 20px;
        margin-bottom: 10px;
    }
    .metric-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #888888;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="sub-header">Upload an image to identify butterfly species</p>', unsafe_allow_html=True)
    
    # Initialize the classifier
    classifier = ButterflyClassifierApp()
    
    # Create sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a neural network model "
        "with hyperparameter tuning to classify butterfly species. "
        "Upload an image of a butterfly to get a species prediction."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        "1. Upload a butterfly image using the file uploader\n"
        "2. Wait for the prediction results\n"
        "3. View the predicted species and confidence level"
    )
    
    # Create two columns for image and results
    col1, col2 = st.columns([2, 1])
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a butterfly image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert PIL Image to numpy array for processing
        img_array = np.array(image)
        
        # Make prediction
        with st.spinner('Analyzing image...'):
            try:
                species, confidence = classifier.predict(img_array)
                
                # Display results
                col2.markdown('<p class="sub-header">Prediction Results:</p>', unsafe_allow_html=True)
                
                col2.markdown('<div class="metric-container">', unsafe_allow_html=True)
                col2.metric("Species", species)
                col2.metric("Confidence", f"{confidence:.2%}")
                col2.markdown('</div>', unsafe_allow_html=True)
                
                # Show confidence bar
                if confidence > 0.7:
                    bar_color = "green"
                elif confidence > 0.5:
                    bar_color = "orange"
                else:
                    bar_color = "red"
                
                col2.progress(float(confidence))
                
                if confidence < 0.5:
                    col2.warning("Low confidence prediction. Consider uploading a clearer image.")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Try uploading a different image.")
    
    else:
        # Display sample info when no image is uploaded
        st.info("Please upload an image of a butterfly to get started.")
        
        # Add sample images for demonstration (optional)
        # st.markdown("### Sample Images")
        # You could add some sample butterfly images here in a grid
    
    # Footer
    st.markdown('<div class="footer">© 2025 Butterfly Classifier App by D3</div>', unsafe_allow_html=True)
    
    # Additional information about the model (expandable)
    with st.expander("Model Information"):
        st.write("""
        This butterfly classifier uses a neural network trained with hyperparameter tuning. 
        It extracts multiple types of features:
        
        - HOG (Histogram of Oriented Gradients) for shape features
        - Color histograms in HSV color space
        - GLCM (Gray Level Co-occurrence Matrix) texture features
        
        These hybrid features are fed into a neural network with dropout and regularization
        for robust classification.
        """)

if __name__ == "__main__":
    main()
