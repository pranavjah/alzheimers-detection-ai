import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .model import create_model
from .feature_extraction import TextFeatureExtractor, AudioFeatureExtractor

def load_model(model_path, metadata_path):
    """Load a trained model."""
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

def evaluate_model(model, text_features, audio_features, labels):
    """Evaluate model performance."""
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i in range(len(text_features)):
            text = torch.FloatTensor(text_features[i:i+1])
            audio = torch.FloatTensor(audio_features[i:i+1])
            
            output = model(text, audio)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            predictions.append(pred.item())
            probabilities.append(prob[0, 1].item())
    
    # Calculate metrics
    report = classification_report(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr, roc_auc)
    }

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_evaluation_results(results, output_dir):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save classification report
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(results['classification_report'])
    
    # Save plots
    plot_confusion_matrix(
        results['confusion_matrix'],
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    fpr, tpr, roc_auc = results['roc_curve']
    plot_roc_curve(
        fpr, tpr, roc_auc,
        save_path=output_dir / 'roc_curve.png'
    ) 