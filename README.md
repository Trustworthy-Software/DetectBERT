# DetectBERT

DetectBERT: Towards Full App-Level Representation Learning to Detect Android Malware

## Environment Setup

To replicate our experiments and use DetectBERT, ensure your environment meets the following requirements:

- **Java**: 11.0.11
- **Python**: 3.7.11
- **Libraries**:
  - numpy: 1.21.6
  - torch: 1.12.1
  - torchvision: 0.2.2
  - torchmetrics: 0.3.2
  - tensorboard: 2.9.1
  - nystrom_attention: 0.0.11
  - scikit-learn: 1.0.2

## Data Preparation
Before training the model, class-level DexBERT embeddings for APKs must be generated:

```bash
cd data
python GenDexBertEmbeddings.py
```
## Model Training and Evaluation

To train and evaluate DetectBERT:

1. Configure the aggregation method and hyperparameters in `model/config.yaml`.
2. Execute the training script:

```bash
cd model
python main.py
```
