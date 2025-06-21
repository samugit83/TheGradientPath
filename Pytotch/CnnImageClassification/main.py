#!/usr/bin/env python3
"""
Fashion-MNIST CNN Classification
Complete training and validation script for Fashion-MNIST dataset using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import time
import os

# Set random seed for reproducibility
torch.manual_seed(0)

# Configuration
IMAGE_SIZE = 16
BATCH_SIZE = 100
LEARNING_RATE = 0.1
NUM_EPOCHS = 5
NUM_CLASSES = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CNN_batch(nn.Module):
    """Convolutional Neural Network with Batch Normalization"""
    
    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN_batch, self).__init__()
        
        # First convolutional layer
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional layer
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)
        self.bn_fc1 = nn.BatchNorm1d(number_of_classes)
    
    def forward(self, x):
        # First conv block
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        
        # Second conv block
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        
        return x

def show_data_samples(dataset, num_samples=3):
    """Display sample images from the dataset"""
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for i in range(num_samples):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze().numpy(), cmap='gray')
        axes[i].set_title(f'Label: {label} ({class_names[label]})')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_data.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("Sample data images saved to 'sample_data.png'")

def create_data_loaders():
    """Create training and validation data loaders"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    print("Loading Fashion-MNIST dataset...")
    train_dataset = datasets.FashionMNIST(
        root='./fashion/data',
        train=True,
        transform=transform,
        download=True
    )
    
    val_dataset = datasets.FashionMNIST(
        root='./fashion/data',
        train=False,
        transform=transform,
        download=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Show sample data
    show_data_samples(val_dataset)
    
    return train_loader, val_loader, len(val_dataset)

def train_model(model, train_loader, val_loader, n_test, criterion, optimizer):
    """Train the CNN model"""
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    
    cost_list = []
    accuracy_list = []
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        epoch_cost = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_cost += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        accuracy = correct / n_test
        avg_cost = epoch_cost / num_batches
        
        cost_list.append(avg_cost)
        accuracy_list.append(accuracy)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_cost:.4f}, Validation Accuracy: {accuracy:.4f}')
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return cost_list, accuracy_list

def plot_results(cost_list, accuracy_list):
    """Plot training cost and validation accuracy"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.plot(range(1, len(cost_list) + 1), cost_list, color=color, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(range(1, len(accuracy_list) + 1), accuracy_list, color=color, marker='s')
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Loss and Validation Accuracy')
    fig.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("Training results plot saved to 'training_results.png'")
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Final Training Loss: {cost_list[-1]:.4f}")
    print(f"Final Validation Accuracy: {accuracy_list[-1]:.4f} ({accuracy_list[-1]*100:.2f}%)")

def save_model(model, filepath='fashion_mnist_cnn.pth'):
    """Save the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'out_1': 16,
            'out_2': 32,
            'number_of_classes': NUM_CLASSES
        }
    }, filepath)
    print(f"Model saved to {filepath}")

def main():
    """Main function to run the complete training pipeline"""
    print("Fashion-MNIST CNN Classification")
    print("=" * 50)
    
    # Create data loaders
    train_loader, val_loader, n_test = create_data_loaders()
    
    # Initialize model
    model = CNN_batch(out_1=16, out_2=32, number_of_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    cost_list, accuracy_list = train_model(
        model, train_loader, val_loader, n_test, criterion, optimizer
    )
    
    # Plot results
    plot_results(cost_list, accuracy_list)
    
    # Save the model
    save_model(model)
    
    print("\nTraining completed successfully!")
    print("Generated files:")
    print("- sample_data.png: Sample images from the dataset")
    print("- training_results.png: Training loss and validation accuracy plot")
    print("- fashion_mnist_cnn.pth: Trained model checkpoint")

if __name__ == "__main__":
    main() 