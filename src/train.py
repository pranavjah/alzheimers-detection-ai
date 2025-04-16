import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from .config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MODELS_DIR
)
from .model import create_model
from .feature_extraction import TextFeatureExtractor, AudioFeatureExtractor

class AlzheimerDataset(Dataset):
    def __init__(self, text_features, audio_features, labels):
        self.text_features = torch.FloatTensor(text_features)
        self.audio_features = torch.FloatTensor(audio_features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'text': self.text_features[idx],
            'audio': self.audio_features[idx],
            'label': self.labels[idx]
        }

def train_model(text_features, audio_features, labels, model_path=None):
    """Train the Alzheimer's detection model."""
    # Split data
    X_train_text, X_test_text, X_train_audio, X_test_audio, y_train, y_test = train_test_split(
        text_features, audio_features, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = AlzheimerDataset(X_train_text, X_train_audio, y_train)
    test_dataset = AlzheimerDataset(X_test_text, X_test_audio, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = create_model(
        text_dim=text_features.shape[1],
        audio_dim=audio_features.shape[1]
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch['text'], batch['audio'])
            loss = criterion(outputs, batch['label'])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch['text'], batch['audio'])
                loss = criterion(outputs, batch['label'])
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch['label'].size(0)
                correct += predicted.eq(batch['label']).sum().item()
        
        # Print progress
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(test_loader):.4f}')
        print(f'Val Accuracy: {100.*correct/total:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss and model_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print('Saved best model!')
    
    return model

def save_model_metadata(model, text_dim, audio_dim, model_path):
    """Save model metadata."""
    metadata = {
        'text_dim': text_dim,
        'audio_dim': audio_dim,
        'hidden_dim': model.text_branch[0].out_features,
        'dropout': model.text_branch[2].p
    }
    
    metadata_path = model_path.parent / f'{model_path.stem}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f) 