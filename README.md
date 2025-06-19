# Bird, Dog, or Reptile? Transfer Learning for Hierarchical Image Classification

This project implements a model for hierarchical multi-label image classification with novelty detection. Our model classifies animal images at two levels: broad super-classes (bird, dog, reptile) and specific sub-classes (species/breeds), while detecting novel categories not seen during training.

To read the full paper, click here: [Read the paper](NNDL_Final_proj.pdf)

## Architecture

### Two-Headed Neural Network
- **Shared Backbone**: Pre-trained ResNet-50 (truncated before final layer)
- **Super-class Head**: Linear classifier for broad categories + novel class
- **Sub-class Head**: Linear classifier for specific categories + novel class

### Novelty Detection
Threshold-based approach using maximum softmax probabilities:
- Super-class threshold (τ_super = 0.7)
- Sub-class threshold (τ_sub = 0.5)


## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- Google Colab (recommended) or local GPU setup

### Dataset Setup for Google Colab
1. Add the `Released_Data_NNDL_2025` folder as a shortcut to your Google Drive's "My Drive"
2. The notebook expects data at: `/content/drive/MyDrive/Released_Data_NNDL_2025`

### Usage
1. **Open in Colab**: Click the Colab badge above or open `NNDL_Final_Proj.ipynb` in Google Colab
2. **Connect to Drive**: Ensure the dataset folder is correctly linked
3. **Run the Notebook**: Execute cells sequentially to train and evaluate the model

The pipeline includes:
- Data loading and preprocessing with augmentation
- Model training with cosine annealing scheduler
- Validation and novelty detection
- Test prediction and submission file generation
