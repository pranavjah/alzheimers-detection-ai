import torch
import torch.nn as nn
from .config import HIDDEN_DIM, DROPOUT

class AlzheimerClassifier(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        
        # Text processing branch
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Audio processing branch
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
    def forward(self, text_features, audio_features):
        # Process text features
        text_out = self.text_branch(text_features)
        
        # Process audio features
        audio_out = self.audio_branch(audio_features)
        
        # Combine features
        combined = torch.cat([text_out, audio_out], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output

def create_model(text_dim, audio_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
    """Create and initialize the model."""
    model = AlzheimerClassifier(
        text_dim=text_dim,
        audio_dim=audio_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model 