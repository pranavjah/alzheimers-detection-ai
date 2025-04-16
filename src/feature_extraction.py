import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from .config import VOCAB_SIZE, MAX_TEXT_LENGTH

class TextFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=VOCAB_SIZE,
            stop_words='english'
        )
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
    def extract_tfidf_features(self, texts):
        """Extract TF-IDF features from texts."""
        return self.vectorizer.fit_transform(texts).toarray()
    
    def extract_bert_features(self, texts):
        """Extract BERT embeddings from texts."""
        features = []
        for text in texts:
            # Tokenize and truncate
            tokens = self.tokenizer(
                text,
                max_length=MAX_TEXT_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.model(**tokens)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            features.append(embedding[0])
        
        return np.array(features)

class AudioFeatureExtractor:
    def __init__(self):
        self.mfcc_features = None
    
    def extract_mfcc_features(self, audio_path):
        """Extract MFCC features from audio file."""
        from .preprocessing import preprocess_audio
        return preprocess_audio(audio_path)
    
    def extract_mel_spectrogram(self, audio_path):
        """Extract Mel spectrogram features from audio file."""
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

def combine_features(text_features, audio_features):
    """Combine text and audio features."""
    return np.concatenate([text_features, audio_features], axis=1) 