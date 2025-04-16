import streamlit as st
import torch
import numpy as np
from pathlib import Path
import json
import librosa
import soundfile as sf
import tempfile
import os

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model
from src.feature_extraction import TextFeatureExtractor, AudioFeatureExtractor
from src.preprocessing import preprocess_audio, tokenize_text

# Page config
st.set_page_config(
    page_title="Alzheimer's Early Detection",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Alzheimer's Early Detection")
st.markdown("""
This application uses machine learning to analyze speech patterns and text for early signs of Alzheimer's disease.
Upload an audio recording and its transcription to get a prediction.
""")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
This tool is for research purposes only and should not be used as a diagnostic tool.
Always consult with healthcare professionals for medical advice.
""")

# Load model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "models" / "best_model.pt"
    metadata_path = Path(__file__).parent.parent / "models" / "best_model_metadata.json"
    
    if not (model_path.exists() and metadata_path.exists()):
        st.error("Model files not found. Please train the model first.")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model = create_model(
        text_dim=metadata['text_dim'],
        audio_dim=metadata['audio_dim'],
        hidden_dim=metadata['hidden_dim'],
        dropout=metadata['dropout']
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

# Initialize feature extractors
@st.cache_resource
def init_feature_extractors():
    return TextFeatureExtractor(), AudioFeatureExtractor()

# Main content
def main():
    # Load model and feature extractors
    model = load_model()
    if model is None:
        return
    
    text_extractor, audio_extractor = init_feature_extractors()
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Audio Recording")
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3'],
            help="Upload a speech recording in WAV or MP3 format"
        )
    
    with col2:
        st.subheader("Enter Transcription")
        text_input = st.text_area(
            "Enter the transcription of the speech",
            height=200,
            help="Type or paste the transcription of the speech recording"
        )
    
    # Process button
    if st.button("Analyze", type="primary"):
        if audio_file is None or not text_input:
            st.error("Please upload an audio file and enter its transcription.")
            return
        
        # Process audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name
        
        try:
            # Extract features
            audio_features = audio_extractor.extract_mfcc_features(audio_path)
            text_features = text_extractor.extract_bert_features([text_input])
            
            # Make prediction
            with torch.no_grad():
                text_tensor = torch.FloatTensor(text_features)
                audio_tensor = torch.FloatTensor([audio_features])
                
                output = model(text_tensor, audio_tensor)
                prob = torch.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
                confidence = prob[0, prediction].item()
            
            # Display results
            st.subheader("Results")
            
            if prediction == 1:
                st.error(f"⚠️ High risk of Alzheimer's detected (Confidence: {confidence:.2%})")
                st.info("""
                This result indicates potential signs of Alzheimer's disease in the speech patterns.
                Please consult with a healthcare professional for proper evaluation.
                """)
            else:
                st.success(f"✅ Low risk of Alzheimer's detected (Confidence: {confidence:.2%})")
                st.info("""
                The analysis suggests typical speech patterns.
                However, this is not a definitive diagnosis.
                Regular check-ups with healthcare professionals are recommended.
                """)
            
            # Display confidence scores
            st.subheader("Confidence Scores")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Low Risk Score",
                    f"{prob[0, 0].item():.2%}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "High Risk Score",
                    f"{prob[0, 1].item():.2%}",
                    delta=None
                )
        
        except Exception as e:
            st.error(f"Error processing the files: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(audio_path)

if __name__ == "__main__":
    main() 