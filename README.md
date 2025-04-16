# Alzheimer's Early Detection AI

An AI-powered system for early detection of Alzheimer's disease through speech and text analysis. This project uses Natural Language Processing (NLP) and speech processing techniques to identify linguistic and acoustic biomarkers indicative of cognitive decline.

## 🧠 Overview

This project aims to develop an AI-driven system that can detect early signs of Alzheimer's disease by analyzing spoken and written language. By examining changes in speech patterns, sentence structure, vocabulary usage, and other linguistic features, the system can help identify potential cognitive decline at an early stage.

### Key Features
- Audio feature extraction using MFCC and Mel spectrograms
- Text analysis using BERT embeddings
- Deep learning model combining audio and text features
- Streamlit web interface for easy interaction
- Comprehensive evaluation metrics and visualizations

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pranavjah/alzheimers_detection_ai.git
cd alzheimers_detection_ai
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🏗️ Project Structure

```
alzheimers_detection_ai/
├── data/
│   ├── raw/              # Raw audio and transcript data
│   └── processed/        # Preprocessed features
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── models/               # Trained models and checkpoints
├── src/
│   ├── __init__.py
│   ├── config.py         # Paths, constants
│   ├── preprocessing.py  # Text and audio preprocessing
│   ├── feature_extraction.py  # Extract linguistic and acoustic features
│   ├── model.py          # Model architectures
│   ├── train.py          # Training loop
│   └── evaluate.py       # Evaluation metrics
├── ui/
│   └── app.py            # Streamlit interface
├── requirements.txt
└── README.md
```

## 🎯 Usage

### Data Preparation

1. Place your training data in the `data/raw` directory:
   - Audio files (WAV or MP3 format)
   - Corresponding text transcriptions

### Training the Model

1. Run the training script:
```bash
python -m src.train
```

2. The trained model will be saved in the `models` directory.

### Evaluation

1. Run the evaluation script:
```bash
python -m src.evaluate
```

2. Evaluation results will be saved in the `models/evaluation` directory.

### Web Interface

1. Start the Streamlit app:
```bash
streamlit run ui/app.py
```

2. Open your web browser and navigate to the provided URL.

3. Use the interface to:
   - Upload audio recordings
   - Enter transcriptions
   - Get predictions with confidence scores
   - View detailed results

## 🔬 Methodology

### Feature Extraction

- **Text Features**:
  - Sentence complexity
  - Grammatical errors
  - Lexical diversity
  - Word frequency
  - BERT embeddings

- **Speech Features**:
  - MFCC (Mel-frequency cepstral coefficients)
  - Mel spectrograms
  - Speech rate
  - Pause duration
  - Pitch variation

### Model Architecture

The model combines both audio and text features through a multi-branch neural network:

1. **Text Branch**: Processes BERT embeddings
2. **Audio Branch**: Processes MFCC features
3. **Combined Classifier**: Makes final predictions

## 📊 Evaluation Metrics

The system provides:
- Classification report
- Confusion matrix
- ROC curve and AUC score
- Confidence scores for predictions

## 📚 References

- [DementiaBank](https://dementia.talkbank.org/)
- [ADReSS Challenge Data](https://www.isca-speech.org/archive/interspeech_2020/abstracts/1177.html)