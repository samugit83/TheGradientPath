"""
AUTOENCODER MODULE FOR NETWORK ATTACK PREDICTION
===============================================

This module implements an AutoEncoder (AE) neural network for feature learning and anomaly detection.

WHAT IS AN AUTOENCODER?
----------------------
An AutoEncoder is a special type of neural network that learns to:
1. **Compress** input data into a smaller representation (encoding)
2. **Reconstruct** the original data from this compressed form (decoding)

The key insight: if the network can accurately reconstruct "normal" data but struggles 
with "abnormal" data, the reconstruction error can indicate anomalies (like network attacks).

ARCHITECTURE:
------------
Input Layer → Hidden Layer (compressed) → Output Layer (reconstructed)
   |              |                          |
 Features      Compressed               Reconstructed
(d_in dims)   Representation              Features
             (d_hidden dims)            (d_in dims)

COMPONENTS:
----------
1. AE: The core neural network architecture
2. AEConfig: Configuration dataclass for hyperparameters  
3. AEWrapper: High-level interface that handles training, inference, and device management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional, Iterable

class AE(nn.Module):
    """
    Simple fully connected AutoEncoder neural network.
    
    ARCHITECTURE EXPLANATION:
    ========================
    This implements a basic "hourglass" shaped neural network:
    
    Input Features (d_in) → Encoder → Hidden Layer (d_hidden) → Decoder → Output (d_in)
         |                     ↓            |                    ↓           |
    Original Data        Compression    Compressed         Reconstruction  Reconstructed
    (e.g., 100 dims)        Layer       Representation        Layer        Data
                                       (e.g., 32 dims)                   (100 dims)
    
    The network learns to compress high-dimensional data into a lower-dimensional 
    representation and then reconstruct it back to the original dimensions.
    
    Parameters:
    -----------
    d_in : int
        Input dimension (number of features in the network traffic data)
    d_hidden : int  
        Hidden dimension (size of the compressed representation)
        Usually much smaller than d_in to force compression
    """
    
    def __init__(self, d_in: int, d_hidden: int):
        """
        Initialize the AutoEncoder architecture.
        
        This constructor builds the neural network layers that will learn
        to compress and reconstruct network traffic data.
        
        Args:
            d_in: Number of input features (e.g., network traffic measurements)
            d_hidden: Size of compressed representation (bottleneck layer)
        """
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU()
        )
        
        self.dec = nn.Sequential(
            nn.Linear(d_hidden, d_in)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compress input data and then reconstruct it.
        
        This is the main computation of the AutoEncoder:
        1. Pass input through encoder to get compressed representation
        2. Pass compressed representation through decoder to get reconstruction
        
        The goal is to make the output as similar as possible to the input.
        The difference (reconstruction error) tells us how "normal" the data is.
        
        Args:
            x: Input tensor of shape (batch_size, d_in) or just (d_in,)
               Contains network traffic features for one or more samples
               
        Returns:
            torch.Tensor: Reconstructed data of same shape as input
                         Should ideally be very close to the original input
        
        MATHEMATICAL FLOW:
        -----------------
        Input x → Encoder → h = ReLU(W₁x + b₁) → Decoder → output = W₂h + b₂
        where:
        - W₁, b₁: encoder weights and biases
        - h: hidden (compressed) representation  
        - W₂, b₂: decoder weights and biases
        """
        encoded = self.enc(x)
        reconstructed = self.dec(encoded)
        return reconstructed

@dataclass
class AEConfig:
    """
    Configuration dataclass for AutoEncoder hyperparameters.
    
    WHAT IS A DATACLASS?
    ===================
    A dataclass is a Python decorator that automatically generates common methods
    like __init__, __repr__, etc. It's perfect for storing configuration parameters.
    
    HYPERPARAMETERS EXPLANATION:
    ===========================
    These parameters control how the AutoEncoder behaves and learns:
    
    1. **d_in**: Input dimension - number of network traffic features
       - Determines the size of input and output layers
       - Must match the number of features in your dataset
       
    2. **d_hidden**: Hidden dimension - size of compressed representation
       - Controls how much the data is compressed
       - Smaller values = more compression but potential information loss
       - Larger values = less compression but may not learn useful patterns
       - Rule of thumb: 1/2 to 1/4 of input dimension
       
    3. **lr**: Learning rate - how fast the network learns
       - Controls the size of weight updates during training
       - Too high: network may not converge (oscillate)
       - Too low: training takes very long
       - 1e-4 (0.0001) is a good starting point for most problems
    
    Usage Example:
    -------------
    config = AEConfig(d_in=100, d_hidden=32, lr=1e-4)
    autoencoder = AEWrapper(config)
    """
    
    d_in: int
    d_hidden: int = 32
    lr: float = 1e-4

class AEWrapper:
    """
    High-level wrapper for the AutoEncoder that handles all the complex details.
    
    WHY DO WE NEED A WRAPPER?
    ========================
    The raw AE class only defines the neural network architecture. To actually USE it,
    we need to handle many additional concerns:
    
    1. **Optimization**: How the network learns from its mistakes (gradient descent)
    2. **Device Management**: Whether to use CPU or GPU for computation
    3. **Training Mode**: Switching between training and evaluation modes
    4. **Gradient Management**: When to compute gradients and when to skip them
    5. **Model Persistence**: Saving and loading trained models
    
    This wrapper encapsulates all these concerns, providing a simple interface:
    - forward_no_grad(x): Get reconstruction without training
    - train_step(x): Train the network on one sample
    - save(path): Save the trained model
    - load(path): Load a pre-trained model
    
    INTERFACE DESIGN PRINCIPLE:
    ==========================
    The wrapper hides complexity and provides only what the user needs.
    You don't need to worry about optimizers, devices, or gradient computation.
    """
    
    def __init__(self, cfg: AEConfig):
        """
        Initialize the AutoEncoder wrapper with all necessary components.
        
        This constructor sets up everything needed for training and inference:
        1. Creates the neural network model
        2. Sets up the optimizer for learning
        3. Configures device (CPU/GPU) for computation
        4. Moves model to appropriate device
        
        Args:
            cfg: AEConfig object containing hyperparameters
                 Must specify d_in (input dimension) at minimum
        """
        
        self.model = AE(cfg.d_in, cfg.d_hidden)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

    def forward_no_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform inference (prediction) without computing gradients.
        
        WHEN TO USE THIS METHOD:
        =======================
        Use this when you want to get the AutoEncoder's reconstruction of input data
        but you're NOT training the network. This is used for:
        
        1. **Feature Selection**: Computing reconstruction errors for ORC
        2. **Anomaly Detection**: Checking how well the network reconstructs new data
        3. **Evaluation**: Testing the model's performance on validation data
        4. **Production**: Using the trained model for real-time prediction
        
        WHY "NO GRAD"?
        =============
        Gradient computation is expensive and only needed during training.
        By disabling gradients, we:
        - Save memory (gradients can double memory usage)
        - Speed up computation (no backward pass calculations)
        - Prevent accidental weight updates
        
        EVALUATION MODE:
        ===============
        Setting model to eval() mode disables training-specific behaviors like:
        - Dropout (if we had any): would set all neurons to active
        - Batch normalization (if we had any): uses running stats instead of batch stats
        
        Args:
            x: Input tensor containing network traffic features
               Shape: (features,) for single sample or (batch_size, features) for batch
               
        Returns:
            torch.Tensor: Reconstructed features on CPU (same shape as input)
                         High reconstruction error suggests anomalous/attack traffic
        """
        
        self.model.eval()
        
        with torch.no_grad():
            input_on_device = x.to(self._device)
            output_on_device = self.model(input_on_device)
            return output_on_device.cpu()

    def train_step(self, x: torch.Tensor) -> float:
        """
        Perform one training step: forward pass, loss computation, and weight update.
        
        WHAT IS A TRAINING STEP?
        =======================
        This is the core of machine learning - how the network learns from its mistakes:
        
        1. **Forward Pass**: Get the network's current prediction (reconstruction)
        2. **Loss Computation**: Measure how wrong the prediction is
        3. **Backward Pass**: Calculate how to adjust weights to reduce error
        4. **Weight Update**: Actually adjust the weights based on gradients
        
        ONLINE LEARNING:
        ===============
        This method implements "online learning" - learning from one sample at a time.
        This is perfect for streaming data where we can't store all samples in memory.
        Each new network traffic sample teaches the model a little bit more.
        
        MSE LOSS EXPLAINED:
        ==================
        We use Mean Squared Error (MSE) loss:
        MSE = (1/n) * Σ(predicted - actual)²
        
        This penalizes large errors more than small ones, encouraging the network
        to minimize reconstruction errors for "normal" network traffic patterns.
        
        Args:
            x: Input tensor containing network traffic features for training
               Shape: (features,) for single sample
               Should represent "normal" network traffic for best results
               
        Returns:
            float: Training loss value (reconstruction error)
                  Lower values indicate better reconstruction ability
        """
        
        self.model.train()
        x = x.to(self._device)
        recon = self.model(x)
        loss = nn.functional.mse_loss(recon, x)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.model.eval()
        return float(loss.item())

    def save(self, path: str):
        """
        Save the trained AutoEncoder model to disk.
        
        WHAT GETS SAVED?
        ===============
        This saves the model's "state_dict" - a Python dictionary containing:
        - All learned weights and biases from encoder and decoder layers
        - Layer structure information (though architecture must be known to load)
        
        What is NOT saved:
        - The optimizer state (Adam's internal momentum/variance estimates)
        - The model architecture definition (you need the code to recreate it)
        - Training history or loss values
        
        FILE FORMAT:
        ===========
        PyTorch uses its own binary format (.pt or .pth files) that efficiently
        stores tensor data with compression. These files are:
        - Compact: Only essential parameters are stored
        - Fast: Optimized for loading back into PyTorch models
        - Portable: Can be loaded on different devices (CPU/GPU)
        
        WHEN TO SAVE:
        ============
        - After training is complete (to use the model later)
        - Periodically during training (checkpointing)
        - When you achieve good performance (backup before experimenting)
        
        Args:
            path: File path where to save the model (e.g., "models/autoencoder.pt")
                 Directory will be created automatically if it doesn't exist
        
        Usage Example:
        -------------
        ae_wrapper.save("artifacts/ae.pt")  # Save to artifacts directory
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """
        Load a previously trained AutoEncoder model from disk.
        
        LOADING PROCESS:
        ===============
        1. Load the state_dict (parameter dictionary) from file
        2. Load parameters into the current model architecture
        3. Set model to evaluation mode for inference
        
        IMPORTANT REQUIREMENTS:
        ======================
        For loading to work correctly:
        - Model architecture must EXACTLY match the saved model
        - Same d_in and d_hidden dimensions as when saved
        - File must exist and be a valid PyTorch state_dict
        
        DEVICE COMPATIBILITY:
        ====================
        The map_location parameter ensures the model loads correctly regardless
        of where it was trained vs where it's being loaded:
        - Trained on GPU, loading on CPU: ✓ Works
        - Trained on CPU, loading on GPU: ✓ Works  
        - Different GPU devices: ✓ Works
        
        POST-LOADING STATE:
        ==================
        After loading:
        - All weights and biases are restored to saved values
        - Model is in evaluation mode (ready for inference)
        - No further training setup needed (optimizer is separate)
        
        Args:
            path: File path to the saved model (e.g., "artifacts/ae.pt")
                 Must be a valid PyTorch state_dict file
        
        Raises:
            FileNotFoundError: If the specified path doesn't exist
            RuntimeError: If state_dict doesn't match model architecture
        
        Usage Example:
        -------------
        ae_wrapper = AEWrapper(config)
        ae_wrapper.load("artifacts/ae.pt")  # Load pre-trained weights
        reconstruction = ae_wrapper.forward_no_grad(data)  # Ready to use!
        """
        state_dict = torch.load(path, map_location=self._device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
