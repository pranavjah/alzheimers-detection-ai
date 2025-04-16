import re
import numpy as np
import librosa
import soundfile as sf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from pathlib import Path
from .config import SAMPLE_RATE, MAX_AUDIO_LENGTH, N_MFCC

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Clean and normalize text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """Tokenize text and remove stopwords."""
    # Clean text first
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def preprocess_audio(audio_path, target_length=MAX_AUDIO_LENGTH):
    """Preprocess audio file to extract MFCC features."""
    # Load audio file
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Trim or pad audio to target length
    if len(y) > target_length * sr:
        y = y[:target_length * sr]
    else:
        y = np.pad(y, (0, max(0, target_length * sr - len(y))))
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Take mean across time
    mfcc_mean = np.mean(mfcc, axis=1)
    
    return mfcc_mean

def save_processed_audio(audio_path, output_path):
    """Save preprocessed audio file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    sf.write(output_path, y, sr)

def process_text_file(text_path):
    """Process text file and return cleaned tokens."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return tokenize_text(text) 