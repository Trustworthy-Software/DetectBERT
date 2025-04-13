# DetectBERT Model

This directory contains the implementation of DetectBERT, a model for Android malware detection using DexBERT embeddings.

## Directory Structure

```
model/
├── main.py              # Main training and evaluation script
├── detect.py            # Script for malware detection using trained model
├── config.yaml          # Configuration file for model parameters
├── models/              # Model architecture implementations
│   └── DetectBERT.py    # DetectBERT model implementation
└── utils/               # Utility functions and helpers
```

## Scripts

### main.py

The main script for training and evaluating the DetectBERT model. It handles:

1. Loading and preprocessing DexBERT embeddings
2. Training the DetectBERT model
3. Evaluating model performance
4. Saving model checkpoints

#### Usage

```bash
python main.py
```

### detect.py

Script for performing malware detection using a trained DetectBERT model. It:

1. Loads a trained model
2. Processes APK embeddings
3. Performs malware detection
4. Generates detailed detection reports

#### Usage

```bash
python detect.py
```

## Model Architecture

DetectBERT is built on top of DexBERT embeddings and includes:

- Input processing for class-level embeddings
- Aggregation mechanisms for app-level representation
- Classification head for malware detection

## Configuration

Model parameters and training settings are configured in `config.yaml`:

- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Evaluation metrics

## Requirements

- Python 3.7.11
- PyTorch 1.12.1
- Other dependencies (see main README)
- Trained DexBERT model
- Generated APK embeddings 