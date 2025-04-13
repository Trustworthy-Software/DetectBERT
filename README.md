# DetectBERT: Towards Full App-Level Representation Learning to Detect Android Malware

DetectBERT is a deep learning-based approach for Android malware detection that leverages DexBERT embeddings to learn full app-level representations.

## Overview

This project implements a novel approach to Android malware detection by:

1. Using DexBERT to generate class-level embeddings from APK bytecode
2. Aggregating these embeddings to create app-level representations
3. Training a classifier to detect malware based on these representations

## Project Structure

```
DetectBERT/
├── data/                # Data processing and embedding generation
│   ├── GenDexBertEmbeddings.py
│   └── SmaliPreprocess.py
├── model/              # DetectBERT implementation
│   ├── main.py
│   ├── detect.py
│   ├── config.yaml
│   └── models/
└── README.md
```

## Environment Setup

### Prerequisites

- **Java**: 11.0.11
- **Python**: 3.7.11
- **CUDA**: 11.3 (for GPU acceleration)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:

- numpy: 1.21.6
- torch: 1.12.1
- torchvision: 0.2.2
- torchmetrics: 0.3.2
- tensorboard: 2.9.1
- nystrom_attention: 0.0.11
- scikit-learn: 1.0.2

## Usage

### 1. Data Preparation

First, generate DexBERT embeddings for your APKs:

```bash
cd data
python GenDexBertEmbeddings.py
```

This will:

- Process APKs listed in source files
- Generate embeddings for each class
- Save embeddings as pickle files

### 2. Model Training

Configure the model in `model/config.yaml`, then train:

```bash
cd model
python main.py
```

The training process will:

- Load and preprocess embeddings
- Train the DetectBERT model
- Save checkpoints and evaluation metrics

### 3. Malware Detection

To detect malware in new APKs:

```bash
cd model
python detect.py
```

This will:

- Load a trained model
- Process APK embeddings
- Generate a detailed detection report

## Configuration

Key configuration files:

- `model/config.yaml`: Model architecture and training parameters
- Source files in `data/`: Lists of APKs to process

## Output

The detection process generates:

- Model checkpoints during training
- Evaluation metrics and TensorBoard logs
- Detailed detection reports with confidence scores

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{sun2024detectbert,
  title={DetectBERT: Towards Full App-Level Representation Learning to Detect Android Malware},
  author={Sun, Tiezhu and Daoudi, Nadia and Kim, Kisub and Allix, Kevin and Bissyand{\'e}, Tegawend{\'e} F and Klein, Jacques},
  booktitle={Proceedings of the 18th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement},
  pages={420--426},
  year={2024}
}
```
