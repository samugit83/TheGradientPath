# Fashion-MNIST CNN Classification

A PyTorch implementation of a Convolutional Neural Network (CNN) for classifying Fashion-MNIST dataset. This project demonstrates image classification using deep learning techniques with batch normalization and proper training/validation procedures.

## Overview

The Fashion-MNIST dataset consists of 70,000 grayscale images of 10 different clothing categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Each image is 28x28 pixels, and in this implementation, we resize them to 16x16 for faster training.

## Model Architecture

The CNN model includes:
- **First Conv Block**: Conv2d(1→16) + BatchNorm + ReLU + MaxPool
- **Second Conv Block**: Conv2d(16→32) + BatchNorm + ReLU + MaxPool
- **Fully Connected**: Linear(512→10) + BatchNorm
- **Output**: 10 classes (Fashion-MNIST categories)

Key Features:
- Batch normalization for better training stability
- ReLU activation functions
- Max pooling for dimensionality reduction
- Cross-entropy loss for multi-class classification
- SGD optimizer with learning rate 0.1

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- pillow

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main script to start training:

```bash
python main.py
```

The script will:
1. Download the Fashion-MNIST dataset automatically
2. Display sample images from the dataset
3. Initialize the CNN model with batch normalization
4. Train for 5 epochs with validation
5. Plot training loss and validation accuracy
6. Save the trained model

### Training Parameters

- **Image Size**: 16x16 pixels
- **Batch Size**: 100
- **Learning Rate**: 0.1
- **Epochs**: 5
- **Optimizer**: SGD
- **Loss Function**: Cross-Entropy

### Output Files

The script generates:
- `sample_data.png`: Sample images from the dataset
- `training_results.png`: Training loss and validation accuracy plot
- `fashion_mnist_cnn.pth`: Trained model checkpoint

## Model Performance

The model typically achieves:
- Training convergence within 5 epochs
- Validation accuracy of ~85-90%
- Fast training due to reduced image size (16x16)

## Project Structure

```
.
├── main.py              # Main training script
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── fashion/            # Dataset directory (created automatically)
├── sample_data.png     # Generated sample images
├── training_results.png # Generated training plots
└── fashion_mnist_cnn.pth # Saved model checkpoint
```

## Key Features

### Data Preprocessing
- Automatic dataset download
- Image resizing to 16x16 pixels
- Tensor conversion with proper normalization

### Model Architecture
- Convolutional layers with batch normalization
- Proper forward pass implementation
- GPU support (if available)

### Training Pipeline
- Proper train/validation split
- Batch processing with DataLoader
- Real-time training progress monitoring
- Model checkpointing

### Visualization
- Sample data visualization
- Training metrics plotting
- Results visualization

## Customization

You can modify the following parameters in `main.py`:

```python
IMAGE_SIZE = 16        # Image dimensions
BATCH_SIZE = 100       # Batch size for training
LEARNING_RATE = 0.1    # Learning rate for SGD
NUM_EPOCHS = 5         # Number of training epochs
```

## GPU Support

The script automatically detects and uses GPU if available:
- CUDA-enabled GPU will be used automatically
- Falls back to CPU if GPU is not available

## Dataset Information

**Fashion-MNIST** is a dataset of Zalando's article images consisting of:
- 60,000 training examples
- 10,000 test examples
- 10 classes
- 28x28 grayscale images

Original dataset: https://github.com/zalandoresearch/fashion-mnist

## License

This project is based on educational material and is intended for learning purposes.

## Acknowledgments

- Original Fashion-MNIST dataset by Zalando Research
- PyTorch community for excellent deep learning framework
- Educational content adapted from IBM's deep learning course materials

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow training**: Enable GPU or reduce image size
3. **Dependencies issues**: Make sure all packages are installed correctly

### Performance Tips

- Use GPU for faster training
- Increase batch size if you have more memory
- Adjust learning rate based on convergence behavior
- Monitor validation accuracy to avoid overfitting

## Results Interpretation

The training script provides:
- Real-time loss and accuracy monitoring
- Final model performance metrics
- Visual plots of training progress
- Saved model for future use

Expected results:
- Training loss should decrease over epochs
- Validation accuracy should improve and stabilize
- Model should achieve reasonable classification performance 