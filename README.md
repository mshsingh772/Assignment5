# MNIST Classifier

A PyTorch implementation of a lightweight CNN for MNIST digit classification that achieves >95% accuracy in a single epoch with less than 25,000 parameters.

## Model Architecture

The model uses a lightweight CNN architecture optimized for quick learning:

### Design Choices

1. **Convolutional Layers**:
   - First layer: 8 filters to capture basic features
   - Second layer: 16 filters for more complex patterns
   - Each followed by batch normalization and max pooling

2. **Batch Normalization**:
   - Added after each conv layer
   - Helps with faster convergence
   - Stabilizes training

3. **Parameter Efficiency**:
   - Minimal number of filters (8→16)
   - Single fully connected layer
   - Total parameters: ~9,146

## Hyperparameters

1. **Optimizer**: SGD with Nesterov Momentum
   - Learning rate: 0.1
   - Momentum: 0.9
   - Nesterov: True


3. **Data Preprocessing**:
   - Normalization: mean=0.1307, std=0.3081 (MNIST statistics)
   - No data augmentation

## Why These Choices?

1. **Minimal Architecture**:
   - Progressive feature extraction (8→16 filters)
   - Batch normalization for stable training
   - Max pooling for spatial dimension reduction

2. **SGD with Momentum**:
   - Better generalization in short training periods
   - Nesterov momentum for improved convergence


3. **Parameter Count**:
   - First conv layer: 80 parameters
   - First batch norm: 16 parameters
   - Second conv layer: 1,168 parameters
   - Second batch norm: 32 parameters
   - Final FC layer: 7,850 parameters
   - Total: 9,146 parameters

## Requirements

- Python 3.8+
- PyTorch
- torchvision


## GitHub Actions

The repository includes automated testing via GitHub Actions to verify:
- Model has less than 25,000 parameters
- Achieves ≥95% training accuracy in one epoch
