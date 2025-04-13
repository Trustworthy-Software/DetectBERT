# Data Processing and Embedding Generation

This directory contains scripts and utilities for processing Android APKs and generating DexBERT embeddings.

## Directory Structure

```
data/
├── GenDexBertEmbeddings.py    # Main script for generating DexBERT embeddings
├── SmaliPreprocess.py         # Utilities for preprocessing Smali code
└── [APK data directories]     # Directories containing APK files and their embeddings
```

## Scripts

### GenDexBertEmbeddings.py

This script processes Android APKs and generates DexBERT embeddings. It performs the following steps:

1. Downloads or copies APK files
2. Disassembles APKs into Smali code
3. Preprocesses Smali code into text format
4. Generates DexBERT embeddings for each class
5. Saves embeddings as pickle files

(Download the pre-trained DexBERT model with this link: [https://drive.google.com/file/d/1z6aZQXT1dS6wX1JgPnWJVS_e6Td2sBPg/view?usp=sharing](https://drive.google.com/file/d/1z6aZQXT1dS6wX1JgPnWJVS_e6Td2sBPg/view?usp=sharing))

#### Usage

```bash
python GenDexBertEmbeddings.py
```

The script processes APKs listed in the source files (e.g., `Infinix_apk.txt`, `Tecno_apk.txt`, `itel_apk.txt`) and generates embeddings in corresponding directories.

### SmaliPreprocess.py

Contains utilities for preprocessing Smali code into a format suitable for DexBERT.

## Data Organization

- APK files and their embeddings are organized in separate directories
- Each APK's embeddings are stored as pickle files
- Source files (e.g., `Infinix_apk.txt`) contain lists of APKs to process

## Requirements

- Java 11.0.11
- Python 3.7.11
- Required Python packages (see main README)
- Sufficient disk space for APK files and embeddings
