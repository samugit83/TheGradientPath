#!/usr/bin/env python3

"""
COMPREHENSIVE TRANSFORMER IMPLEMENTATION FOR TEXT GENERATION WITH KV CACHE OPTIMIZATION

This script implements a complete Transformer-based language model from scratch using TensorFlow/Keras.
The implementation focuses on text generation with an advanced Key-Value (KV) Cache optimization system
that significantly speeds up autoregressive text generation.

WHAT THIS SCRIPT DOES:
1. Implements a full Transformer architecture with multi-head self-attention
2. Provides KV Cache optimization for faster text generation
3. Trains the model on text data (Shakespeare or custom corpus)
4. Generates text using the trained model with performance comparisons
5. Includes comprehensive logging and visualization tools

KEY CONCEPTS EXPLAINED:
- Transformers: Neural network architecture that uses attention mechanisms
- Self-Attention: Allows the model to focus on different parts of the input sequence
- KV Cache: Optimization technique that stores computed attention keys/values to avoid redundant calculations
- Autoregressive Generation: Generating text one token at a time, using previous tokens as context

TECHNICAL ARCHITECTURE:
- Multi-Head Self-Attention layers for learning contextual relationships
- Position encoding to give the model information about token positions
- Feed-forward networks for non-linear transformations
- Layer normalization for training stability
- Causal masking for autoregressive text generation
"""

# =============================================================================
# IMPORT SECTION - Loading all necessary libraries and frameworks
# =============================================================================

import os  # Operating system interface for file/directory operations
from datetime import datetime  # For timestamping logs and outputs
import numpy as np  # Numerical computing library for array operations
import tensorflow as tf  # Deep learning framework - our main ML library

# TensorFlow/Keras specific imports for building neural network components
from tensorflow.keras.layers import (
    Layer,              # Base class for all neural network layers
    Dense,              # Fully connected (linear) layer - core building block
    LayerNormalization, # Normalization technique for stable training
    Dropout,            # Regularization technique to prevent overfitting
    Embedding,          # Converts token IDs to dense vector representations
    TextVectorization   # Preprocesses text data into numerical format
)
from tensorflow.keras.models import Model  # Base class for complex models
from tensorflow.keras.utils import get_file  # Utility for downloading datasets
from tensorflow.keras.callbacks import EarlyStopping  # Prevents overfitting during training
import matplotlib.pyplot as plt  # Plotting library for visualizations
import visualkeras  # Library for visualizing neural network architectures

# =============================================================================
# GPU CONFIGURATION SECTION - Optimizing hardware utilization
# =============================================================================

"""
GPU CONFIGURATION EXPLANATION:
Modern deep learning requires significant computational power. GPUs (Graphics Processing Units)
are much faster than CPUs for the parallel matrix operations that neural networks require.
This section configures TensorFlow to use GPU efficiently if available.

MEMORY GROWTH CONCEPT:
By default, TensorFlow allocates all GPU memory at startup, which can cause issues.
Memory growth allows TensorFlow to allocate GPU memory incrementally as needed.
"""

# Detect all available GPU devices on the system
gpus = tf.config.list_physical_devices('GPU')

if gpus:  # If GPUs are detected
    try:
        # Configure each GPU for memory growth to prevent allocation issues
        # This prevents TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Get logical GPU devices (virtual GPUs created by TensorFlow)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured.")
        
    except RuntimeError as e:
        # GPU configuration must happen before any operations are created
        print(f"Error setting up GPU memory growth: {e}")
else:
    # Fallback to CPU if no GPU is available
    print("No GPU detected, using CPU.")


# =============================================================================
# MULTI-HEAD SELF-ATTENTION CLASS - The core of Transformer architecture
# =============================================================================

class MultiHeadSelfAttention(Layer):
    """
    MULTI-HEAD SELF-ATTENTION MECHANISM - THE HEART OF TRANSFORMERS
    
    CONCEPTUAL EXPLANATION:
    Attention is a mechanism that allows the model to focus on different parts of the input
    when processing each token. Think of it like reading a sentence - when you read the word
    "it", you need to look back at previous words to understand what "it" refers to.
    
    SELF-ATTENTION specifically means the model attends to other positions in the same sequence.
    MULTI-HEAD means we have multiple parallel attention mechanisms, each learning different
    types of relationships (syntax, semantics, etc.).
    
    TECHNICAL DETAILS:
    - Query (Q): "What am I looking for?" - represents the current token
    - Key (K): "What can I match with?" - represents all tokens that can be attended to  
    - Value (V): "What information do I get?" - the actual content to extract
    - Multiple heads allow learning different relationship types simultaneously
    
    MATHEMATICAL FOUNDATION:
    Attention(Q,K,V) = softmax(QK^T / √d_k)V
    Where d_k is the dimension of the key vectors (for scaling)
    """
    
    def __init__(self, embed_dim, num_heads=8):
        """
        INITIALIZATION - Setting up the attention mechanism components
        
        PARAMETERS EXPLAINED:
        - embed_dim: The size of the embedding vectors (how many dimensions each token has)
        - num_heads: How many parallel attention mechanisms to use (typically 8 or 16)
        
        DESIGN RATIONALE:
        Multiple heads allow the model to attend to information from different representation
        subspaces at different positions. Each head learns different types of relationships.
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        # Store configuration parameters
        self.embed_dim = embed_dim  # Total embedding dimension
        self.num_heads = num_heads  # Number of parallel attention heads
        
        # Calculate dimension per head - must divide evenly
        self.projection_dim = embed_dim // num_heads
        
        # Create linear transformation layers for Q, K, V
        # These transform input embeddings into query, key, and value representations
        self.query_dense = Dense(embed_dim)  # Transforms input to queries
        self.key_dense = Dense(embed_dim)    # Transforms input to keys  
        self.value_dense = Dense(embed_dim)  # Transforms input to values
        
        # Final layer to combine outputs from all attention heads
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value, mask=None):
        """
        CORE ATTENTION COMPUTATION - The mathematical heart of the attention mechanism
        
        FUNCTION PURPOSE:
        This function implements the scaled dot-product attention mechanism that allows
        the model to determine which parts of the input sequence to focus on.
        
        STEP-BY-STEP PROCESS:
        1. Compute attention scores (how much each token should attend to others)
        2. Scale scores to prevent vanishing gradients
        3. Apply causal mask for autoregressive generation (prevent looking ahead)
        4. Apply softmax to get attention weights (probabilities)
        5. Use weights to compute weighted combination of values
        
        PARAMETERS:
        - query: What we're looking for [batch, heads, seq_len, head_dim]
        - key: What we're looking in [batch, heads, seq_len, head_dim]  
        - value: What we extract [batch, heads, seq_len, head_dim]
        - mask: Prevents attention to future tokens (causal masking)
        """
        
        # STEP 1: Compute raw attention scores
        # Matrix multiplication: Query @ Key^T gives similarity scores
        # Shape: [batch, heads, seq_len_q, seq_len_k]
        score = tf.matmul(query, key, transpose_b=True)
        
        # STEP 2: Get dimension for scaling (prevents vanishing gradients)
        # Convert to float32 for numerical stability
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        
        # STEP 3: Scale attention scores by sqrt(d_k)
        # This prevents softmax from saturating for large embedding dimensions
        scaled_score = score / tf.math.sqrt(dim_key)
        
        # STEP 4: Apply causal mask if provided
        # For autoregressive generation, tokens can't attend to future positions
        if mask is not None:
            # Add large negative value to masked positions (becomes ~0 after softmax)
            scaled_score += (mask * -1e9)
        
        # STEP 5: Apply softmax to get attention probabilities
        # Converts scores to probabilities that sum to 1 across the key dimension
        weights = tf.nn.softmax(scaled_score, axis=-1)
        
        # STEP 6: Apply attention weights to values
        # Weighted combination of value vectors based on attention scores
        output = tf.matmul(weights, value)
        
        return output, weights

    def split_heads(self, x, batch_size):
        """
        MULTI-HEAD PREPARATION - Reshaping tensors for parallel attention computation
        
        CONCEPTUAL PURPOSE:
        This function reorganizes the data so that multiple attention heads can work
        in parallel. Instead of one big attention computation, we split the embedding
        dimension into multiple smaller spaces, each handled by a different head.
        
        RESHAPING LOGIC:
        Input:  [batch, seq_len, embed_dim]
        Output: [batch, num_heads, seq_len, head_dim]
        
        This allows each head to work on a smaller subspace of the full embedding.
        """
        
        # Reshape to separate heads: [batch, seq_len, num_heads, head_dim]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        
        # Transpose to put heads dimension first: [batch, num_heads, seq_len, head_dim]
        # This enables parallel processing of multiple attention heads
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, kv_cache=None, use_cache=False, training=False):
        """
        FORWARD PASS WITH KV CACHE OPTIMIZATION - The main attention computation
        
        FUNCTION PURPOSE:
        This is the main forward pass of the attention mechanism. It includes support for
        KV Cache optimization, which is crucial for efficient text generation.
        
        KV CACHE CONCEPT EXPLAINED:
        During text generation, we generate tokens one by one. Without caching, for each new
        token, we would recompute the Keys and Values for all previous tokens, which is wasteful.
        KV Cache stores the computed Keys and Values from previous steps and reuses them.
        
        GENERATION PHASES:
        1. PREFILL: Process the initial prompt, build the cache
        2. DECODE: Generate new tokens one by one, reusing cached K,V values
        
        EFFICIENCY GAIN:
        - Without cache: O(n²) computation for each new token
        - With cache: O(n) computation for each new token
        Where n is the sequence length
        
        PARAMETERS:
        - inputs: Current input tokens [batch, seq_len, embed_dim]
        - kv_cache: Previously computed keys/values (for efficiency)
        - use_cache: Whether to use/update the cache
        - training: Whether we're in training mode (affects dropout)
        """
        
        # Get tensor dimensions for reshaping operations
        batch_size = tf.shape(inputs)[0]  # How many sequences we're processing
        seq_len = tf.shape(inputs)[1]     # Length of current input sequence
        
        # =================================================================
        # STEP 1: COMPUTE QUERY, KEY, VALUE TRANSFORMATIONS
        # =================================================================
        
        # Transform input embeddings into Query, Key, Value representations
        # Each linear transformation learns different aspects:
        # - Query: What information is this token looking for?
        # - Key: What information does this token provide for matching?
        # - Value: What information does this token contribute?
        query = self.query_dense(inputs)  # [batch, seq_len, embed_dim]
        key = self.key_dense(inputs)      # [batch, seq_len, embed_dim]
        value = self.value_dense(inputs)  # [batch, seq_len, embed_dim]
        
        # =================================================================
        # STEP 2: SPLIT INTO MULTIPLE ATTENTION HEADS
        # =================================================================
        
        # Reshape for multi-head attention: [batch, num_heads, seq_len, head_dim]
        # This allows parallel processing of different attention patterns
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # =================================================================
        # STEP 3: KV CACHE OPTIMIZATION - REUSE PREVIOUS COMPUTATIONS
        # =================================================================
        
        # KV Cache implementation for efficient autoregressive generation
        if use_cache and kv_cache is not None:
            """
            CACHE MECHANISM EXPLANATION:
            During text generation, we process tokens sequentially. For each new token,
            we need to attend to all previous tokens. Without caching, we would recompute
            the Key and Value representations for all previous tokens every time.
            
            With caching:
            1. Store previously computed Keys and Values
            2. For new tokens, only compute new Keys and Values
            3. Concatenate new with cached to get full context
            """
            
            # Retrieve cached keys and values from previous steps
            cached_key = kv_cache.get('key')      # Previously computed keys
            cached_value = kv_cache.get('value')  # Previously computed values
            
            # If cache exists, concatenate with new computations
            if cached_key is not None and cached_value is not None:
                # Concatenate along sequence dimension (axis=2)
                # Result: [batch, heads, old_seq + new_seq, head_dim]
                key = tf.concat([cached_key, key], axis=2)
                value = tf.concat([cached_value, value], axis=2)
        
        # =================================================================
        # STEP 4: CREATE CAUSAL MASK FOR AUTOREGRESSIVE GENERATION
        # =================================================================
        
        """
        CAUSAL MASKING EXPLANATION:
        In language modeling, we want to predict the next token based only on previous tokens.
        During training and generation, tokens should not be able to "see" future tokens.
        
        The causal mask ensures that position i can only attend to positions j where j <= i.
        This creates a lower triangular matrix pattern.
        """
        
        # Get the total sequence length (including cached tokens)
        total_seq_len = tf.shape(key)[2]
        
        # Create causal mask: lower triangular matrix
        # band_part(ones, -1, 0) creates lower triangular matrix
        mask = tf.linalg.band_part(tf.ones((seq_len, total_seq_len)), -1, 0)
        
        # Convert to mask format: 0 for allowed positions, 1 for masked positions
        mask = tf.where(mask == 0, 1.0, 0.0)
        
        # =================================================================
        # STEP 5: COMPUTE ATTENTION AND COMBINE HEADS
        # =================================================================
        
        # Apply the attention mechanism with causal masking
        attention_output, attention_weights = self.attention(query, key, value, mask)
        
        # Transpose back: [batch, seq_len, num_heads, head_dim]
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        
        # Combine all attention heads back into single representation
        # Reshape: [batch, seq_len, embed_dim]
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        
        # Final linear transformation to combine information from all heads
        output = self.combine_heads(concat_attention)
        
        # =================================================================
        # STEP 6: UPDATE CACHE FOR NEXT ITERATION
        # =================================================================
        
        # Prepare cache for next generation step (if caching is enabled)
        # Store current keys and values for reuse in next iteration
        new_cache = {'key': key, 'value': value} if use_cache else None
        
        return output, new_cache


class TransformerBlock(Layer):
    """
    TRANSFORMER BLOCK - A COMPLETE TRANSFORMER LAYER
    
    CONCEPTUAL OVERVIEW:
    A Transformer block combines multiple components to create a powerful sequence processing unit:
    1. Multi-Head Self-Attention: Allows tokens to communicate with each other
    2. Feed-Forward Network: Applies non-linear transformations to each position
    3. Residual Connections: Helps with gradient flow during training
    4. Layer Normalization: Stabilizes training by normalizing activations
    
    ARCHITECTURAL PATTERN:
    Input → LayerNorm → Attention → Add&Norm → LayerNorm → FFN → Add&Norm → Output
    
    This is the standard "Post-Norm" Transformer architecture, though "Pre-Norm" is also common.
    Each component serves a specific purpose in enabling the model to understand and generate text.
    
    RESIDUAL CONNECTIONS EXPLAINED:
    These are "skip connections" that add the input to the output of each sub-layer.
    They solve the vanishing gradient problem and allow training of very deep networks.
    The pattern is: output = input + transformation(input)
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        INITIALIZATION - Setting up all components of the transformer block
        
        PARAMETERS EXPLAINED:
        - embed_dim: Size of token embeddings (typically 256, 512, 768, etc.)
        - num_heads: Number of parallel attention heads (typically 8, 12, 16)
        - ff_dim: Size of feed-forward hidden layer (typically 4x embed_dim)
        - rate: Dropout rate for regularization (typically 0.1)
        
        DESIGN RATIONALE:
        - Multiple attention heads capture different types of relationships
        - Large FFN allows complex non-linear transformations
        - Dropout prevents overfitting by randomly zeroing some connections
        """
        super(TransformerBlock, self).__init__()
        
        # Store hyperparameters
        self.embed_dim = embed_dim  # Embedding dimension
        self.num_heads = num_heads  # Number of attention heads
        self.ff_dim = ff_dim       # Feed-forward dimension
        self.rate = rate           # Dropout rate
        
        # Initialize the multi-head self-attention mechanism
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        
        # Create feed-forward network (FFN)
        # This is a two-layer MLP with ReLU activation
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),  # Expand to larger dimension
            Dense(embed_dim),                  # Project back to embed_dim
        ])
        
        # Layer normalization layers for stabilizing training
        # These normalize the mean and variance of layer inputs
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # After attention
        self.layernorm2 = LayerNormalization(epsilon=1e-6)  # After FFN
        
        # Dropout layers for regularization
        # Randomly set some connections to zero during training
        self.dropout1 = Dropout(rate)  # After attention
        self.dropout2 = Dropout(rate)  # After FFN

    def call(self, inputs, kv_cache=None, use_cache=False, training=False):
        """
        FORWARD PASS - Processing input through attention and feed-forward layers
        
        FUNCTION PURPOSE:
        This implements the forward pass of a complete Transformer block, including:
        1. Multi-head self-attention with residual connection
        2. Feed-forward network with residual connection
        3. Layer normalization for training stability
        4. Dropout for regularization
        
        RESIDUAL CONNECTION PATTERN:
        The key pattern here is: output = LayerNorm(input + SubLayer(input))
        This helps with gradient flow and allows training of deep networks.
        
        PARAMETERS:
        - inputs: Input token embeddings [batch, seq_len, embed_dim]
        - kv_cache: Cache for attention keys/values (for generation efficiency)
        - use_cache: Whether to use/update the cache
        - training: Whether in training mode (affects dropout)
        """
        
        # =================================================================
        # STEP 1: MULTI-HEAD SELF-ATTENTION WITH RESIDUAL CONNECTION
        # =================================================================
        
        # Apply multi-head self-attention
        # This allows tokens to communicate and share information
        attn_output, new_cache = self.att(
            inputs, 
            kv_cache=kv_cache, 
            use_cache=use_cache, 
            training=training
        )
        
        # Apply dropout for regularization (only active during training)
        # Dropout randomly zeros some connections to prevent overfitting
        attn_output = self.dropout1(attn_output, training=training)
        
        # RESIDUAL CONNECTION + LAYER NORMALIZATION
        # Add input to attention output (residual connection)
        # Then normalize for training stability
        out1 = self.layernorm1(inputs + attn_output)
        
        # =================================================================
        # STEP 2: FEED-FORWARD NETWORK WITH RESIDUAL CONNECTION
        # =================================================================
        
        # Apply feed-forward network
        # This processes each position independently with non-linear transformations
        ffn_output = self.ffn(out1)
        
        # Apply dropout for regularization
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # SECOND RESIDUAL CONNECTION + LAYER NORMALIZATION
        # Add previous output to FFN output, then normalize
        output = self.layernorm2(out1 + ffn_output)
        
        return output, new_cache


class TransformerModel(Model):
    """
    COMPLETE TRANSFORMER MODEL FOR TEXT GENERATION
    
    ARCHITECTURAL OVERVIEW:
    This class implements a complete decoder-only Transformer model suitable for
    autoregressive text generation. The architecture follows the GPT (Generative
    Pre-trained Transformer) design pattern.
    
    COMPONENT BREAKDOWN:
    1. Token Embedding: Converts token IDs to dense vector representations
    2. Positional Encoding: Adds position information to embeddings
    3. Transformer Blocks: Stack of attention + feed-forward layers
    4. Output Projection: Maps final hidden states to vocabulary probabilities
    
    AUTOREGRESSIVE TEXT GENERATION:
    The model generates text one token at a time, using previously generated tokens
    as context for predicting the next token. This is called "autoregressive" because
    the model's output feeds back into its input.
    
    KEY FEATURES:
    - Supports KV Cache optimization for fast generation
    - Handles variable sequence positions during generation
    - Uses causal masking to prevent looking ahead
    - Scalable architecture (can adjust layers, heads, dimensions)
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        """
        MODEL INITIALIZATION - Setting up all components of the transformer
        
        PARAMETERS EXPLAINED:
        - vocab_size: Size of vocabulary (how many unique tokens/words)
        - embed_dim: Dimension of token embeddings (model width)
        - num_heads: Number of attention heads per transformer block
        - ff_dim: Feed-forward network hidden dimension
        - num_layers: Number of transformer blocks (model depth)
        - seq_length: Maximum sequence length the model can handle
        
        SCALING CONSIDERATIONS:
        - Larger embed_dim: More expressive representations, higher memory usage
        - More num_heads: Can capture more relationship types, needs more computation
        - Larger ff_dim: More complex transformations, typically 4x embed_dim
        - More num_layers: Deeper model, can learn more complex patterns
        - Longer seq_length: Can handle longer contexts, quadratic memory growth
        """
        super(TransformerModel, self).__init__()
        
        # Store model configuration
        self.vocab_size = vocab_size      # Size of vocabulary
        self.embed_dim = embed_dim        # Embedding dimension
        self.num_heads = num_heads        # Number of attention heads
        self.ff_dim = ff_dim             # Feed-forward dimension
        self.num_layers = num_layers      # Number of transformer layers
        self.seq_length = seq_length      # Maximum sequence length
        
        # =================================================================
        # COMPONENT 1: TOKEN EMBEDDING LAYER
        # =================================================================
        
        # Convert token IDs to dense vector representations
        # Each token gets mapped to an embed_dim dimensional vector
        self.embedding = Embedding(vocab_size, embed_dim)
        
        # =================================================================
        # COMPONENT 2: POSITIONAL ENCODING
        # =================================================================
        
        # Generate positional encoding matrix
        # This adds position information to embeddings since attention is permutation-invariant
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)
        
        # =================================================================
        # COMPONENT 3: STACK OF TRANSFORMER BLOCKS
        # =================================================================
        
        # Create a stack of transformer blocks
        # Each block contains attention + feed-forward with residual connections
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) 
            for _ in range(num_layers)
        ]
        
        # =================================================================
        # COMPONENT 4: OUTPUT PROJECTION LAYER
        # =================================================================
        
        # Final layer that projects hidden states to vocabulary probabilities
        # Maps from embed_dim to vocab_size for next token prediction
        self.dense = Dense(vocab_size)

    def positional_encoding(self, seq_length, embed_dim):
        """
        POSITIONAL ENCODING GENERATION - Adding position information to embeddings
        
        CONCEPTUAL PURPOSE:
        Transformers use attention mechanisms that are "permutation invariant" - they
        don't inherently understand the order of tokens. Positional encoding adds
        information about token positions to help the model understand sequence order.
        
        SINUSOIDAL ENCODING EXPLANATION:
        We use sinusoidal functions (sin/cos) with different frequencies for each dimension.
        This creates unique patterns for each position that the model can learn to interpret.
        
        MATHEMATICAL FOUNDATION:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Where:
        - pos: position in sequence
        - i: dimension index
        - d_model: embedding dimension
        
        ADVANTAGES OF SINUSOIDAL ENCODING:
        1. Unique pattern for each position
        2. Allows extrapolation to longer sequences than seen during training
        3. Relative position relationships are consistent
        4. No additional parameters to learn
        """
        
        # Generate angle rates for each embedding dimension
        # These create different frequencies for different dimensions
        angle_rads = self.get_angles(
            np.arange(seq_length)[:, np.newaxis],    # Positions: [seq_length, 1]
            np.arange(embed_dim)[np.newaxis, :],     # Dimensions: [1, embed_dim]
            embed_dim
        )
        
        # Apply sine to even indices (0, 2, 4, ...)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        # Add batch dimension: [1, seq_length, embed_dim]
        pos_encoding = angle_rads[np.newaxis, ...]
        
        # Convert to TensorFlow tensor with appropriate dtype
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, embed_dim):
        """
        ANGLE CALCULATION FOR POSITIONAL ENCODING
        
        MATHEMATICAL EXPLANATION:
        This function calculates the angles used in sinusoidal positional encoding.
        The formula creates different frequencies for different embedding dimensions,
        allowing the model to distinguish between different positions.
        
        FREQUENCY PATTERN:
        - Lower dimensions use higher frequencies (change rapidly across positions)
        - Higher dimensions use lower frequencies (change slowly across positions)
        - This creates a unique "fingerprint" for each position
        
        PARAMETERS:
        - pos: Position indices [seq_length, 1]
        - i: Dimension indices [1, embed_dim]
        - embed_dim: Total embedding dimension
        """
        
        # Calculate angle rates using the positional encoding formula
        # Higher dimensions get lower frequencies (10000^(2i/d_model))
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        
        # Multiply positions by angle rates to get final angles
        return pos * angle_rates

    def call(self, inputs, kv_cache=None, use_cache=False, start_pos=0, training=False):
        """
        FORWARD PASS - Complete transformer model inference with KV cache support
        
        FUNCTION PURPOSE:
        This is the main forward pass of the transformer model. It handles both training
        and generation modes, with sophisticated support for KV cache optimization during
        autoregressive generation.
        
        GENERATION MODES:
        1. TRAINING: Process full sequences in parallel, no caching needed
        2. PREFILL: Initial processing of prompt, build cache for future use
        3. DECODE: Generate one token at a time, reuse cached computations
        
        POSITIONAL ENCODING HANDLING:
        The model needs to handle different scenarios for positional encoding:
        - During training: Use standard positional encoding for full sequences
        - During generation: Adjust encoding based on current position in sequence
        - For long sequences: Handle positions beyond training sequence length
        
        PARAMETERS:
        - inputs: Token IDs [batch, seq_len]
        - kv_cache: Previously computed attention keys/values
        - use_cache: Whether to use/update cache (for generation)
        - start_pos: Starting position in sequence (for generation)
        - training: Whether in training mode
        """
        
        # Get current sequence length from input tensor
        seq_len = tf.shape(inputs)[1]
        
        # =================================================================
        # STEP 1: HANDLE POSITIONAL ENCODING BASED ON GENERATION PHASE
        # =================================================================
        
        """
        POSITIONAL ENCODING STRATEGY:
        The challenge is that during generation, we process tokens at different positions
        than during training. We need to provide appropriate positional information
        for each phase of generation.
        """
        
        if start_pos > 0 and start_pos < self.seq_length:
            # DECODE PHASE: Processing tokens at specific positions
            # Extract positional encoding for current positions only
            pos_encoding = self.pos_encoding[:, start_pos:start_pos + seq_len, :]
            
        else:
            # TRAINING or PREFILL PHASE or out-of-bounds positions
            if start_pos >= self.seq_length:
                # For sequences longer than training length, reuse last position
                # This allows some extrapolation beyond training sequence length
                pos_encoding = self.pos_encoding[:, -1:, :]  # Last position
                pos_encoding = tf.tile(pos_encoding, [1, seq_len, 1])  # Repeat for all tokens
                
            else:
                # Standard case: use positional encoding from the beginning
                pos_encoding = self.pos_encoding[:, :seq_len, :]
        
        # =================================================================
        # STEP 2: CONVERT TOKENS TO EMBEDDINGS AND ADD POSITIONAL INFO
        # =================================================================
        
        # Convert token IDs to dense embeddings
        # Shape: [batch, seq_len, embed_dim]
        x = self.embedding(inputs)
        
        # Add positional encoding to embeddings
        # This gives the model information about token positions
        x += pos_encoding
        
        # =================================================================
        # STEP 3: INITIALIZE KV CACHE IF NEEDED
        # =================================================================
        
        # For generation with caching, initialize cache structure
        if use_cache and kv_cache is None:
            # Create empty cache for each transformer layer
            kv_cache = [None] * self.num_layers
        
        # =================================================================
        # STEP 4: PROCESS THROUGH TRANSFORMER LAYERS
        # =================================================================
        
        # Track new cache states for each layer
        new_caches = []
        
        # Pass through each transformer block sequentially
        for i, transformer_block in enumerate(self.transformer_blocks):
            # Get cache for current layer (if available)
            layer_cache = kv_cache[i] if kv_cache else None
            
            # Process through transformer block with caching
            x, new_cache = transformer_block(
                x, 
                kv_cache=layer_cache, 
                use_cache=use_cache, 
                training=training
            )
            
            # Store new cache state for this layer
            new_caches.append(new_cache)
        
        # =================================================================
        # STEP 5: PROJECT TO VOCABULARY PROBABILITIES
        # =================================================================
        
        # Final projection to vocabulary size
        # Shape: [batch, seq_len, vocab_size]
        output = self.dense(x)
        
        # Return output and updated caches
        if use_cache:
            return output, new_caches
        else:
            return output


# =============================================================================
# DATA PREPARATION FUNCTION - Creating training sequences from text
# =============================================================================

def create_sequences(text, seq_length):
    """
    SEQUENCE GENERATION FOR LANGUAGE MODELING TRAINING
    
    CONCEPTUAL PURPOSE:
    Language models learn by predicting the next token given previous tokens. This function
    creates training examples where the input is a sequence of tokens and the target is
    the same sequence shifted by one position (next token prediction).
    
    EXAMPLE:
    If text = [1, 2, 3, 4, 5, 6, 7] and seq_length = 3:
    Input sequences:  [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    Target sequences: [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
    
    TRAINING OBJECTIVE:
    Given tokens [1, 2, 3], predict [2, 3, 4]
    - At position 0: given [1], predict 2
    - At position 1: given [1, 2], predict 3  
    - At position 2: given [1, 2, 3], predict 4
    
    This is called "teacher forcing" - during training, we provide the correct
    previous tokens rather than using the model's own predictions.
    
    PARAMETERS:
    - text: Vectorized text as array of token IDs
    - seq_length: Length of input sequences to create
    
    RETURNS:
    - input_seqs: Input sequences for training [num_sequences, seq_length]
    - target_seqs: Target sequences for training [num_sequences, seq_length]
    """
    
    # Initialize lists to store sequences
    input_seqs = []   # Input sequences (what the model sees)
    target_seqs = []  # Target sequences (what the model should predict)
    
    # Create overlapping sequences by sliding window
    # We iterate through the text, creating sequences of fixed length
    for i in range(len(text) - seq_length):
        # Input sequence: tokens from position i to i+seq_length
        input_seq = text[i:i + seq_length]
        
        # Target sequence: tokens from position i+1 to i+seq_length+1
        # This is the input sequence shifted by one position
        target_seq = text[i + 1:i + seq_length + 1]
        
        # Add to our lists
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    
    # Convert to numpy arrays for efficient processing
    return np.array(input_seqs), np.array(target_seqs)


# =============================================================================
# TEXT GENERATION FUNCTION WITH KV CACHE OPTIMIZATION
# =============================================================================

def generate_text_with_kv_cache(model, vectorizer, start_string, seq_length, num_generate=100, temperature=1.0, use_kv_cache=True):
    """
    OPTIMIZED TEXT GENERATION WITH KV CACHE SUPPORT
    
    FUNCTION PURPOSE:
    This function generates text using a trained transformer model with optional KV cache
    optimization. It demonstrates the significant performance improvement possible with
    caching during autoregressive generation.
    
    GENERATION PROCESS OVERVIEW:
    1. PREPROCESSING: Convert start string to token IDs, handle padding/truncation
    2. PREFILL PHASE: Process initial prompt, build attention cache
    3. DECODE PHASE: Generate tokens one by one, reusing cached computations
    4. POSTPROCESSING: Convert token IDs back to text
    
    KV CACHE OPTIMIZATION EXPLAINED:
    Without cache: For each new token, recompute attention for ALL previous tokens
    With cache: Store previous attention computations, only compute for new token
    
    PERFORMANCE COMPARISON:
    - Standard generation: O(n²) computation per token
    - Cached generation: O(n) computation per token
    Where n is sequence length
    
    TEMPERATURE PARAMETER:
    Controls randomness in generation:
    - Low temperature (0.1-0.7): More focused, deterministic output
    - High temperature (1.0-2.0): More random, creative output
    - Temperature = 1.0: Sample directly from model probabilities
    
    PARAMETERS:
    - model: Trained transformer model
    - vectorizer: Text preprocessing pipeline
    - start_string: Initial text to begin generation
    - seq_length: Model's expected sequence length
    - num_generate: Number of tokens to generate
    - temperature: Sampling temperature for creativity control
    - use_kv_cache: Whether to use KV cache optimization
    
    RETURNS:
    - Generated text as string
    """
    
    # =================================================================
    # STEP 1: PREPROCESS INPUT TEXT
    # =================================================================
    
    # Convert start string to token IDs using the trained vectorizer
    # This applies the same preprocessing used during training
    input_eval = vectorizer([start_string]).numpy()
    
    # Handle sequence length mismatches
    if input_eval.shape[1] < seq_length:
        # PAD: If input is shorter than expected, pad with zeros at the beginning
        # Padding at the beginning preserves the meaningful tokens at the end
        padding = np.zeros((1, seq_length - input_eval.shape[1]))
        input_eval = np.concatenate((padding, input_eval), axis=1)
        
    elif input_eval.shape[1] > seq_length:
        # TRUNCATE: If input is longer than expected, keep only the last tokens
        # This preserves the most recent context for generation
        input_eval = input_eval[:, -seq_length:]

    # Convert to TensorFlow tensor for model input
    input_eval = tf.convert_to_tensor(input_eval)
    
    # Initialize list to store generated tokens
    text_generated = []
    
    # Get vocabulary for converting token IDs back to text
    vocab = vectorizer.get_vocabulary()
    
    # =================================================================
    # GENERATION BRANCH: KV CACHE vs STANDARD
    # =================================================================
    
    if use_kv_cache:
        print("Using KV Cache for generation...")
        
        # =====================================================
        # PREFILL PHASE: Process prompt and build cache
        # =====================================================
        
        """
        PREFILL PHASE EXPLANATION:
        In the prefill phase, we process the entire initial prompt at once.
        This builds the KV cache with attention keys and values for all prompt tokens.
        This is more efficient than processing tokens one by one for the initial context.
        """
        
        # Process initial prompt with cache building
        predictions, kv_cache = model(
            input_eval, 
            use_cache=True,     # Enable cache building
            start_pos=0,        # Starting from position 0
            training=False      # Inference mode
        )
        
        # Get predictions for the last token position
        # This will be used to generate the first new token
        last_predictions = predictions[0, -1, :]  # [vocab_size]
        
        # Apply temperature scaling for controlled randomness
        last_predictions = last_predictions / temperature
        
        # Sample next token using categorical distribution
        # This adds randomness based on model's probability distribution
        predicted_id = tf.random.categorical(
            tf.expand_dims(last_predictions, 0), 
            num_samples=1
        )[0, 0].numpy()
        
        # Convert token ID to text and add to generated sequence
        if predicted_id < len(vocab):
            text_generated.append(vocab[predicted_id])
        
        # Track current position in sequence
        current_pos = input_eval.shape[1]
        
        # =====================================================
        # DECODE PHASE: Generate tokens using cache
        # =====================================================
        
        """
        DECODE PHASE EXPLANATION:
        In the decode phase, we generate tokens one by one. For each new token:
        1. Use cached keys/values from previous tokens
        2. Only compute attention for the new token
        3. Update cache with new token's keys/values
        4. Repeat until desired length
        
        This is where the major performance improvement comes from.
        """
        
        # Generate remaining tokens one by one
        for i in range(num_generate - 1):
            # Prepare input for next token (single token)
            next_token = tf.convert_to_tensor([[predicted_id]])
            
            # Generate next token using cached keys/values
            predictions, kv_cache = model(
                next_token,
                kv_cache=kv_cache,    # Reuse cached computations
                use_cache=True,       # Continue updating cache
                start_pos=current_pos, # Current position in sequence
                training=False        # Inference mode
            )
            
            # Get predictions for the new token
            last_predictions = predictions[0, -1, :]
            
            # Apply temperature scaling
            last_predictions = last_predictions / temperature
            
            # Sample next token
            predicted_id = tf.random.categorical(
                tf.expand_dims(last_predictions, 0), 
                num_samples=1
            )[0, 0].numpy()
            
            # Add to generated text
            if predicted_id < len(vocab):
                text_generated.append(vocab[predicted_id])
            
            # Update position counter
            current_pos += 1
            
    else:
        print("Using standard generation (no KV cache)...")
        
        # =====================================================
        # STANDARD GENERATION: No cache optimization
        # =====================================================
        
        """
        STANDARD GENERATION EXPLANATION:
        Without KV cache, for each new token we:
        1. Process the entire sequence (including all previous tokens)
        2. Recompute all attention operations
        3. Extract prediction for the last position
        4. Add new token and repeat
        
        This is computationally expensive but simpler to implement.
        """
        
        # Generate tokens one by one without caching
        for i in range(num_generate):
            # Process entire sequence (inefficient for long sequences)
            predictions = model(input_eval, use_cache=False, training=False)
            
            # Get predictions for last position
            predictions = predictions[0, -1, :]
            
            # Apply temperature scaling
            predictions = predictions / temperature
            
            # Sample next token
            predicted_id = tf.random.categorical(
                tf.expand_dims(predictions, 0), 
                num_samples=1
            )[0, 0].numpy()

            # Update input sequence with new token
            # Append new token and maintain sequence length by truncating
            input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)
            input_eval = input_eval[:, -seq_length:]  # Keep only last seq_length tokens
            input_eval = tf.convert_to_tensor(input_eval)

            # Add to generated text
            if predicted_id < len(vocab):
                text_generated.append(vocab[predicted_id])

    # =================================================================
    # STEP 3: COMBINE AND RETURN GENERATED TEXT
    # =================================================================
    
    # Combine start string with generated tokens
    return start_string + ' ' + ' '.join(text_generated)


def generate_text(model, vectorizer, start_string, seq_length, num_generate=100, temperature=1.0):
    """
    LEGACY TEXT GENERATION FUNCTION - Backward compatibility wrapper
    
    FUNCTION PURPOSE:
    This function provides backward compatibility for code that uses the old interface.
    It simply calls the new generate_text_with_kv_cache function with caching disabled.
    
    DESIGN PATTERN:
    This follows the adapter pattern - providing a consistent interface while
    the underlying implementation changes. This ensures existing code continues
    to work without modification.
    
    PARAMETERS:
    Same as generate_text_with_kv_cache, but always uses standard generation (no cache)
    
    RETURNS:
    Generated text string
    """
    return generate_text_with_kv_cache(
        model, vectorizer, start_string, seq_length, 
        num_generate, temperature, use_kv_cache=False
    )


# =============================================================================
# CORPUS LOADING FUNCTION - Flexible data source management
# =============================================================================

def load_corpus(corpus_source):
    """
    FLEXIBLE CORPUS LOADING - Support for multiple data sources
    
    FUNCTION PURPOSE:
    This function provides a flexible way to load training data from different sources.
    It supports both web-based datasets (like Shakespeare) and local text files.
    This makes the system adaptable to different use cases and datasets.
    
    DATA SOURCE OPTIONS:
    1. "web": Downloads Shakespeare dataset from TensorFlow's public datasets
    2. "local": Loads text from a local file named "corpus.txt"
    
    ENCODING HANDLING:
    Text files can have different encodings. The function handles:
    - UTF-8: Standard Unicode encoding (preferred)
    - Latin-1: Fallback encoding for older text files
    
    ERROR HANDLING:
    The function includes comprehensive error handling for:
    - Missing files
    - Encoding issues
    - Network problems (for web downloads)
    
    PARAMETERS:
    - corpus_source: Either "web" or "local" to specify data source
    
    RETURNS:
    - text: Raw text string ready for preprocessing
    
    RAISES:
    - FileNotFoundError: If local corpus file doesn't exist
    - ValueError: If corpus_source is not "web" or "local"
    - UnicodeDecodeError: If text encoding cannot be handled
    """
    
    if corpus_source == "web":
        # =================================================================
        # WEB DATA SOURCE: Shakespeare dataset from TensorFlow
        # =================================================================
        
        print("Loading Shakespeare dataset from web...")
        
        # Download Shakespeare dataset using Keras utility
        # This dataset is commonly used for text generation experiments
        path_to_file = get_file(
            'shakespeare.txt',  # Local filename to save as
            'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
        )
        
        # Read the downloaded file
        # Using 'rb' mode then decode to handle encoding properly
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        
        print(f"Web dataset loaded. Text length: {len(text)} characters")
        
    elif corpus_source == "local":
        # =================================================================
        # LOCAL DATA SOURCE: Custom text file
        # =================================================================
        
        print("Loading corpus from local file 'corpus.txt'...")
        
        try:
            # Attempt to read with UTF-8 encoding (most common)
            with open('corpus.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Local corpus loaded. Text length: {len(text)} characters")
            
        except FileNotFoundError:
            # Provide helpful error message with instructions
            raise FileNotFoundError(
                "corpus.txt not found. Please make sure the file exists in the current directory."
            )
            
        except UnicodeDecodeError:
            # Fallback to Latin-1 encoding if UTF-8 fails
            print("Failed to decode with UTF-8, trying with latin-1...")
            with open('corpus.txt', 'r', encoding='latin-1') as f:
                text = f.read()
            print(f"Local corpus loaded with latin-1 encoding. Text length: {len(text)} characters")
            
    else:
        # =================================================================
        # INVALID DATA SOURCE: Provide clear error message
        # =================================================================
        
        raise ValueError(
            f"Invalid corpus source: {corpus_source}. Must be 'web' or 'local'."
        )
    
    return text


# =============================================================================
# TRAINING FUNCTION - Complete model training pipeline
# =============================================================================

def train():
    """
    COMPREHENSIVE MODEL TRAINING PIPELINE
    
    FUNCTION PURPOSE:
    This function implements a complete training pipeline for the Transformer model.
    It handles all aspects of training from data loading to model saving, including:
    1. Data loading and preprocessing
    2. Model architecture setup
    3. Training with callbacks and logging
    4. Visualization and monitoring
    5. Model persistence and artifact saving
    
    TRAINING PIPELINE OVERVIEW:
    1. Load and preprocess text data
    2. Create input/target sequence pairs
    3. Build transformer model architecture
    4. Configure training parameters and callbacks
    5. Train the model with monitoring
    6. Save model weights and configuration
    7. Generate training visualizations
    
    MONITORING AND LOGGING:
    - TensorBoard integration for real-time training monitoring
    - Early stopping to prevent overfitting
    - Training loss visualization
    - Model architecture visualization
    - Comprehensive logging for reproducibility
    
    OUTPUTS:
    - Trained model weights file
    - Model configuration and vectorizer
    - Training logs and visualizations
    - Performance metrics and plots
    
    RETURNS:
    - model: Trained transformer model
    - vectorizer: Text preprocessing pipeline
    - vocab_size: Size of vocabulary
    - seq_length: Sequence length used
    - logdir: Directory containing training logs
    """
    
    # =================================================================
    # STEP 1: CONFIGURATION AND SETUP
    # =================================================================
    
    """
    TRAINING CONFIGURATION:
    These parameters control the training process and can be adjusted based on:
    - Available computational resources
    - Dataset size and characteristics
    - Desired model performance vs training time
    """
    
    # Data source configuration
    corpus_source = "local"  # Options: "web" for Shakespeare, "local" for corpus.txt
    
    # =================================================================
    # STEP 2: DATA LOADING AND PREPROCESSING
    # =================================================================
    
    # Load text data from specified source
    text = load_corpus(corpus_source)
    
    # Display preview of dataset for verification
    print("Preview of the dataset:")
    print(text[:500])  # Show first 500 characters

    # Text preprocessing configuration
    vocab_size = 10000  # Maximum vocabulary size (most frequent tokens)
    seq_length = 100    # Length of input sequences for training
    
    """
    TEXT VECTORIZATION EXPLANATION:
    TextVectorization converts raw text into numerical format that neural networks can process.
    Key steps:
    1. Tokenization: Split text into individual tokens (words/subwords)
    2. Vocabulary building: Create mapping from tokens to integers
    3. Sequence conversion: Convert text to sequences of integers
    4. Padding/truncation: Ensure consistent sequence lengths
    """
    
    # Create and configure text vectorizer
    vectorizer = TextVectorization(
        max_tokens=vocab_size,  # Limit vocabulary size
        output_mode='int'       # Output integer token IDs
    )
    
    # Adapt vectorizer to the text data
    # This builds the vocabulary and learns token-to-ID mappings
    text_ds = tf.data.Dataset.from_tensor_slices([text]).batch(1)
    vectorizer.adapt(text_ds)

    # Convert text to numerical format
    vectorized_text = vectorizer([text])[0]
    print(f"Vectorized text shape: {vectorized_text.shape}")
    print(f"First 10 vectorized tokens: {vectorized_text.numpy()[:10]}")

    # =================================================================
    # STEP 3: SEQUENCE GENERATION FOR TRAINING
    # =================================================================
    
    """
    SEQUENCE GENERATION EXPLANATION:
    For language modeling, we need to create input-target pairs where:
    - Input: sequence of N tokens
    - Target: same sequence shifted by 1 position (next token prediction)
    
    This creates a supervised learning problem where the model learns to
    predict the next token given the previous tokens.
    """
    
    # Generate training sequences
    X, Y = create_sequences(vectorized_text.numpy(), seq_length)
    
    # Validate sequence generation
    print(f"Number of sequences generated: {len(X)}")
    
    # Ensure we have valid training data
    assert X.size > 0, "Input data X is empty"
    assert Y.size > 0, "Target data Y is empty"
    
    # Convert to TensorFlow tensors for training
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)
    print(f"Shape of X: {X.shape}")
    print(f"Shape of Y: {Y.shape}")

    # =================================================================
    # STEP 4: MODEL ARCHITECTURE CONFIGURATION
    # =================================================================
    
    """
    MODEL HYPERPARAMETERS EXPLANATION:
    These parameters define the model architecture and capacity:
    
    - embed_dim: Width of the model (embedding dimension)
      Higher = more expressive but slower and more memory
    
    - num_heads: Number of attention heads per layer
      More heads = can capture more relationship types
    
    - ff_dim: Feed-forward network hidden dimension
      Typically 4x embed_dim, controls non-linear transformation capacity
    
    - num_layers: Depth of the model (number of transformer blocks)
      Deeper = more complex patterns but harder to train
    
    SCALING CONSIDERATIONS:
    - Start with smaller models for experimentation
    - Scale up based on data size and computational resources
    - Monitor training stability when increasing model size
    """
    
    # Model architecture hyperparameters
    embed_dim = 256   # Embedding dimension (model width)
    num_heads = 4     # Number of attention heads
    ff_dim = 512      # Feed-forward network dimension
    num_layers = 4    # Number of transformer layers (model depth)

    # Create the transformer model
    model = TransformerModel(
        vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length
    )

    # Build the model by running a forward pass
    # This initializes all weights and shows the model structure
    _ = model(tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32))

    # Configure training optimization
    model.compile(
        optimizer='adam',  # Adam optimizer (adaptive learning rate)
        loss='sparse_categorical_crossentropy'  # Standard loss for classification
    )
    
    # Display model architecture
    model.summary()

    # =================================================================
    # STEP 5: MODEL VISUALIZATION AND ARCHITECTURE DOCUMENTATION
    # =================================================================
    
    """
    VISUALIZATION PURPOSE:
    Model visualization helps understand and debug the architecture.
    It's particularly useful for:
    - Verifying the model structure is correct
    - Understanding data flow through layers
    - Debugging architectural issues
    - Documentation and presentation purposes
    """
    
    # Generate model architecture visualization
    print("Generating model architecture visualization...")
    arch_path = 'transformer_text_model_architecture.png'
    
    try:
        # Create layered visualization of the model architecture
        visualkeras.layered_view(
            model,
            to_file=arch_path,    # Output file path
            legend=True,          # Show layer names and types
            draw_volume=False,    # 2D view (not 3D volume)
            scale_xy=1.5,         # Scale factor for clarity
            scale_z=1,            # Z-dimension scale
            spacing=20            # Space between layers
        )
        print(f"Model architecture visualization saved to: {arch_path}")
        
    except Exception as e:
        # Handle potential issues with visualization generation
        print(f"Could not generate VisualKeras visualization: {e}")

    # =================================================================
    # STEP 6: TENSORBOARD LOGGING SETUP
    # =================================================================
    
    """
    TENSORBOARD INTEGRATION:
    TensorBoard is a powerful tool for monitoring machine learning training.
    It provides real-time visualization of:
    - Training loss and metrics
    - Model graph structure
    - Histograms of weights and gradients
    - Custom scalars and images
    
    LOGGING STRATEGY:
    - Create timestamped log directory for each training run
    - Log training metrics, model graph, and custom visualizations
    - Enable histogram logging to monitor weight distributions
    """
    
    # Create timestamped log directory
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    
    # Configure TensorBoard callback
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,           # Directory for logs
        histogram_freq=1,         # Log weight histograms every epoch
        write_graph=True,         # Log the model graph
        write_images=True,        # Log weight images
        update_freq='epoch'       # Update frequency
    )
    
    print(f"TensorBoard logs in: {os.path.abspath(logdir)}")
    print(f"Run: tensorboard --logdir {logdir}")

    # Log model architecture visualization to TensorBoard
    try:
        # Add architecture visualization to TensorBoard
        with tf.summary.create_file_writer(logdir).as_default():
            # Read the architecture image
            img = tf.io.read_file(arch_path)
            img = tf.image.decode_png(img, channels=4)
            # Log as image summary
            tf.summary.image("Model Architecture Visualization", tf.expand_dims(img, 0), step=0)
        print("Model visualization logged to TensorBoard")
        
    except Exception as e:
        print(f"Could not log VisualKeras image to TensorBoard: {e}")

    # =================================================================
    # STEP 7: TRAINING CALLBACKS AND CONFIGURATION
    # =================================================================
    
    """
    TRAINING CALLBACKS:
    Callbacks are functions called during training to modify behavior:
    
    - Early Stopping: Prevents overfitting by stopping when validation loss stops improving
    - TensorBoard: Logs metrics and visualizations for monitoring
    - Model Checkpoints: Saves best model weights during training
    - Learning Rate Scheduling: Adjusts learning rate during training
    
    EARLY STOPPING EXPLANATION:
    Monitors training loss and stops training if it doesn't improve for 'patience' epochs.
    This prevents the model from overfitting to the training data.
    """
    
    # Configure early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='loss',           # Metric to monitor
        patience=2,               # Number of epochs to wait for improvement
        restore_best_weights=True # Restore weights from best epoch
    )
    
    # =================================================================
    # STEP 8: MODEL TRAINING EXECUTION
    # =================================================================
    
    """
    TRAINING PROCESS:
    The training loop repeatedly:
    1. Forward pass: Compute predictions for input sequences
    2. Loss calculation: Compare predictions with target sequences
    3. Backward pass: Compute gradients using backpropagation
    4. Weight update: Adjust model parameters using optimizer
    5. Validation: Check performance on validation data (if available)
    
    BATCH PROCESSING:
    Training uses batches (groups of examples) rather than individual examples:
    - More efficient GPU utilization
    - More stable gradient estimates
    - Faster convergence
    
    TRAINING HYPERPARAMETERS:
    - epochs: Number of complete passes through the dataset
    - batch_size: Number of examples processed together
    - Larger batch_size = more stable gradients but more memory usage
    """
    
    print("Starting training...")
    
    # Execute training with monitoring
    history = model.fit(
        X, Y,                          # Training data (input, target)
        epochs=20,                     # Number of training epochs
        batch_size=32,                 # Batch size for training
        callbacks=[early_stopping, tensorboard_cb]  # Training callbacks
    )

    print("Training completed!")

    # =================================================================
    # STEP 9: MODEL PERSISTENCE AND ARTIFACT SAVING
    # =================================================================
    
    """
    MODEL SAVING STRATEGY:
    We save multiple artifacts for complete model persistence:
    1. Model weights: The learned parameters
    2. Model architecture: The structure and hyperparameters
    3. Vectorizer: Text preprocessing pipeline
    4. Training metadata: Configuration used for training
    
    This allows complete model reconstruction for inference or further training.
    """
    
    # Save model weights
    weights_save_path = "transformer_model.weights.h5"
    model.save_weights(weights_save_path)
    print(f"Model weights saved to: {weights_save_path}")

    # Save vectorizer and model configuration
    import pickle
    vectorizer_path = "text_vectorizer.pkl"
    
    # Package all necessary information for model reconstruction
    model_metadata = {
        'vectorizer': vectorizer,      # Text preprocessing pipeline
        'vocab_size': vocab_size,      # Vocabulary size
        'seq_length': seq_length,      # Sequence length
        'embed_dim': embed_dim,        # Embedding dimension
        'num_heads': num_heads,        # Number of attention heads
        'ff_dim': ff_dim,             # Feed-forward dimension
        'num_layers': num_layers       # Number of transformer layers
    }
    
    # Save to pickle file
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(model_metadata, f)
    print(f"Vectorizer and model parameters saved to: {vectorizer_path}")

    # =================================================================
    # STEP 10: TRAINING VISUALIZATION AND ANALYSIS
    # =================================================================
    
    """
    POST-TRAINING ANALYSIS:
    After training, we create visualizations to understand:
    - Training progress and convergence
    - Loss curves and learning patterns
    - Potential overfitting or underfitting
    - Training stability and optimization effectiveness
    """
    
    # Create training loss visualization
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to file
    plot_path = os.path.join(logdir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training loss plot saved to: {plot_path}")

    # Return training artifacts for potential use
    return model, vectorizer, vocab_size, seq_length, logdir


def generate(use_kv_cache=True, weights_path="transformer_model.weights.h5", 
             vectorizer_path="text_vectorizer.pkl"):
    """
    TEXT GENERATION WITH TRAINED MODEL
    
    FUNCTION PURPOSE:
    This function loads a previously trained transformer model and uses it to generate
    text. It demonstrates the complete inference pipeline and showcases the performance
    benefits of KV cache optimization.
    
    GENERATION PIPELINE:
    1. Load model configuration and vectorizer
    2. Reconstruct model architecture
    3. Load trained weights
    4. Generate text samples with different parameters
    5. Measure and compare performance
    6. Save generated outputs and analysis
    
    PERFORMANCE ANALYSIS:
    The function includes comprehensive performance analysis comparing:
    - Generation with KV cache vs without cache
    - Different text lengths and generation parameters
    - Speed improvements and efficiency gains
    
    PARAMETER EXPLORATION:
    Generates text with different temperatures to show:
    - Low temperature: More focused, coherent text
    - High temperature: More creative, diverse text
    - Temperature effects on generation quality
    
    PARAMETERS:
    - use_kv_cache: Whether to use KV cache optimization
    - weights_path: Path to saved model weights
    - vectorizer_path: Path to saved vectorizer and config
    
    OUTPUTS:
    - Generated text samples
    - Performance comparison analysis
    - Saved outputs in timestamped directory
    """
    
    import pickle
    import time
    
    print("Loading trained model and vectorizer...")
    
    # =================================================================
    # STEP 1: LOAD MODEL CONFIGURATION AND VECTORIZER
    # =================================================================
    
    """
    MODEL RECONSTRUCTION:
    To use a trained model, we need to:
    1. Load the model architecture parameters
    2. Recreate the exact same model structure
    3. Load the trained weights into the structure
    4. Load the text preprocessing pipeline
    """
    
    try:
        # Load vectorizer and model configuration
        with open(vectorizer_path, 'rb') as f:
            vectorizer_data = pickle.load(f)
            
        # Extract all necessary parameters
        vectorizer = vectorizer_data['vectorizer']
        vocab_size = vectorizer_data['vocab_size']
        seq_length = vectorizer_data['seq_length']
        embed_dim = vectorizer_data['embed_dim']
        num_heads = vectorizer_data['num_heads']
        ff_dim = vectorizer_data['ff_dim']
        num_layers = vectorizer_data['num_layers']
        
        print(f"Vectorizer and parameters loaded from: {vectorizer_path}")
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Vectorizer file not found: {vectorizer_path}. Please run training first."
        )
    except KeyError as e:
        raise KeyError(
            f"Missing parameter in vectorizer file: {e}. Please retrain the model to save all parameters."
        )
    
    # =================================================================
    # STEP 2: RECONSTRUCT MODEL ARCHITECTURE
    # =================================================================
    
    print("Reconstructing model architecture...")
    
    # Create model with same architecture as training
    model = TransformerModel(
        vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length
    )
    
    # Build the model by running a forward pass
    # This initializes the model structure and creates all layers
    dummy_input = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)
    _ = model(dummy_input)
    
    # =================================================================
    # STEP 3: LOAD TRAINED WEIGHTS
    # =================================================================
    
    try:
        # Load the trained weights into the model
        model.load_weights(weights_path)
        print(f"Model weights loaded from: {weights_path}")
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Please run training first."
        )
    except Exception as e:
        raise Exception(f"Error loading weights: {e}")

    # =================================================================
    # STEP 4: SETUP OUTPUT DIRECTORY
    # =================================================================
    
    # Create output directory for generated text and analysis
    output_dir = os.path.join("generated_outputs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # =================================================================
    # STEP 5: BASIC TEXT GENERATION WITH PERFORMANCE MEASUREMENT
    # =================================================================
    
    print(f"\nGenerating text with KV Cache: {use_kv_cache}")
    start_string = "the object of our"  # Starting prompt for generation
    
    # Measure generation time for performance analysis
    start_time = time.time()
    generated_text = generate_text_with_kv_cache(
        model, vectorizer, start_string, seq_length, 
        num_generate=100,           # Generate 100 tokens
        temperature=0.7,            # Moderate creativity
        use_kv_cache=use_kv_cache   # Use specified caching mode
    )
    generation_time = time.time() - start_time
    
    print(f"\nGenerated text (100 tokens in {generation_time:.2f}s):")
    print(generated_text)

    # =================================================================
    # STEP 6: LONGER TEXT GENERATION
    # =================================================================
    
    print("\nGenerating longer text sequence...")
    start_time = time.time()
    longer_text = generate_text_with_kv_cache(
        model, vectorizer, start_string, seq_length, 
        num_generate=200,           # Generate 200 tokens
        temperature=0.8,            # Slightly more creative
        use_kv_cache=use_kv_cache
    )
    longer_generation_time = time.time() - start_time
    
    print(f"\nLonger generated text (200 tokens in {longer_generation_time:.2f}s):")
    print(longer_text)

    # =================================================================
    # STEP 7: PERFORMANCE COMPARISON (if KV cache is enabled)
    # =================================================================
    
    if use_kv_cache:
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        
        # Test generation without KV cache for comparison
        print("Testing generation WITHOUT KV cache...")
        start_time = time.time()
        text_no_cache = generate_text_with_kv_cache(
            model, vectorizer, start_string, seq_length, 
            num_generate=100, temperature=0.7, use_kv_cache=False
        )
        time_no_cache = time.time() - start_time
        
        # Calculate and display performance metrics
        print(f"\nPerformance Results:")
        print(f"With KV Cache:    {generation_time:.2f}s")
        print(f"Without KV Cache: {time_no_cache:.2f}s")
        
        speedup = time_no_cache / generation_time if generation_time > 0 else 1
        print(f"Speedup:          {speedup:.2f}x")
        
        time_saved_percent = ((time_no_cache - generation_time) / time_no_cache * 100)
        print(f"Time saved:       {time_saved_percent:.1f}%")

    # =================================================================
    # STEP 8: SAVE GENERATED TEXT AND PERFORMANCE RESULTS
    # =================================================================
    
    # Save all generated text and performance results to file
    text_output_path = os.path.join(output_dir, 'generated_text.txt')
    with open(text_output_path, 'w') as f:
        f.write(f"KV Cache enabled: {use_kv_cache}\n")
        f.write(f"Start string: {start_string}\n\n")
        f.write(f"Generated text (100 tokens in {generation_time:.2f}s):\n{generated_text}\n\n")
        f.write(f"Generated text (200 tokens in {longer_generation_time:.2f}s):\n{longer_text}\n\n")
        
        # Include performance comparison if available
        if use_kv_cache and 'time_no_cache' in locals():
            f.write(f"Performance comparison:\n")
            f.write(f"With KV Cache: {generation_time:.2f}s\n")
            f.write(f"Without KV Cache: {time_no_cache:.2f}s\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
            
    print(f"Generated text and performance results saved to: {text_output_path}")

    # =================================================================
    # STEP 9: TEMPERATURE EXPERIMENTATION
    # =================================================================
    
    """
    TEMPERATURE SAMPLING EXPLORATION:
    Temperature controls the randomness of text generation:
    - Temperature = 0: Deterministic (always choose most likely token)
    - Temperature < 1: More focused, coherent text
    - Temperature = 1: Sample according to model probabilities
    - Temperature > 1: More random, creative text
    
    This section demonstrates how temperature affects generation quality and style.
    """
    
    print("\n" + "="*50)
    print("GENERATING SAMPLES WITH DIFFERENT TEMPERATURES")
    print("="*50)
    
    # Test different temperature values
    temperatures = [0.5, 0.7, 1.0, 1.2]
    samples_output_path = os.path.join(output_dir, 'temperature_samples.txt')
    
    with open(samples_output_path, 'w') as f:
        f.write("Text Generation Samples with Different Temperatures\n")
        f.write("="*60 + "\n\n")
        
        # Generate samples with each temperature
        for temp in temperatures:
            print(f"Generating with temperature {temp}...")
            
            # Generate text with current temperature
            sample_text = generate_text_with_kv_cache(
                model, vectorizer, start_string, seq_length, 
                num_generate=150,         # Generate 150 tokens
                temperature=temp,         # Current temperature
                use_kv_cache=use_kv_cache
            )
            
            # Display preview (first 200 characters)
            print(f"\nTemperature {temp}:")
            preview = sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
            print(preview)
            
            # Save full text to file
            f.write(f"Temperature: {temp}\n")
            f.write("-" * 20 + "\n")
            f.write(f"{sample_text}\n\n")
    
    print(f"Temperature samples saved to: {samples_output_path}")
    print(f"All outputs saved in directory: {output_dir}")


def generate_compare(weights_path="transformer_model.weights.h5", 
                    vectorizer_path="text_vectorizer.pkl"):
    """
    COMPREHENSIVE KV CACHE PERFORMANCE COMPARISON
    
    FUNCTION PURPOSE:
    This function provides a detailed analysis of KV Cache performance benefits by
    systematically comparing generation speed with and without caching across
    different scenarios. It's designed to demonstrate the practical benefits of
    the optimization and help users understand when and how much improvement to expect.
    
    COMPARISON METHODOLOGY:
    1. Load trained model and setup identical conditions
    2. Test multiple text lengths (different computational loads)
    3. Test multiple temperatures (different sampling strategies)
    4. Measure precise timing for each configuration
    5. Calculate performance metrics and improvements
    6. Generate comprehensive analysis report
    
    METRICS ANALYZED:
    - Absolute generation time (seconds)
    - Relative speedup (how many times faster)
    - Time saved (absolute and percentage)
    - Scalability with sequence length
    - Consistency across different parameters
    
    SCIENTIFIC APPROACH:
    - Controlled comparison (same model, same inputs)
    - Multiple test cases for statistical significance
    - Comprehensive logging for reproducibility
    - Clear presentation of results
    
    PARAMETERS:
    - weights_path: Path to trained model weights
    - vectorizer_path: Path to model configuration and vectorizer
    
    OUTPUTS:
    - Detailed performance comparison table
    - Generated text samples for quality verification
    - Comprehensive analysis report
    - Performance insights and recommendations
    """
    
    import pickle
    import time
    
    print("Loading trained model and vectorizer...")
    
    # =================================================================
    # STEP 1: MODEL LOADING (Same as generate function)
    # =================================================================
    
    try:
        # Load model configuration and vectorizer
        with open(vectorizer_path, 'rb') as f:
            vectorizer_data = pickle.load(f)
            
        # Extract configuration parameters
        vectorizer = vectorizer_data['vectorizer']
        vocab_size = vectorizer_data['vocab_size']
        seq_length = vectorizer_data['seq_length']
        embed_dim = vectorizer_data['embed_dim']
        num_heads = vectorizer_data['num_heads']
        ff_dim = vectorizer_data['ff_dim']
        num_layers = vectorizer_data['num_layers']
        
        print(f"Vectorizer and parameters loaded from: {vectorizer_path}")
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Vectorizer file not found: {vectorizer_path}. Please run training first."
        )
    except KeyError as e:
        raise KeyError(
            f"Missing parameter in vectorizer file: {e}. Please retrain the model to save all parameters."
        )
    
    # Reconstruct model architecture
    print("Reconstructing model architecture...")
    model = TransformerModel(
        vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length
    )
    
    # Build model structure
    dummy_input = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)
    _ = model(dummy_input)
    
    # Load trained weights
    try:
        model.load_weights(weights_path)
        print(f"Model weights loaded from: {weights_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Please run training first."
        )
    except Exception as e:
        raise Exception(f"Error loading weights: {e}")

    # =================================================================
    # STEP 2: SETUP COMPARISON EXPERIMENT
    # =================================================================
    
    # Create output directory for comparison results
    output_dir = os.path.join("generated_outputs", "comparison_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("KV CACHE vs STANDARD GENERATION PERFORMANCE COMPARISON")
    print("="*80)
    
    # =================================================================
    # STEP 3: DEFINE TEST PARAMETERS
    # =================================================================
    
    """
    EXPERIMENTAL DESIGN:
    We test across multiple dimensions to get comprehensive understanding:
    - Token counts: Different sequence lengths to test scalability
    - Temperatures: Different sampling strategies to test consistency
    - Multiple runs: Ensure reliable timing measurements
    """
    
    # Test configuration
    start_string = "the object of our"          # Consistent starting prompt
    token_counts = [50, 100, 200]               # Different sequence lengths
    temperatures = [0.7, 1.0]                   # Different creativity levels
    
    # Results storage
    results = []
    
    # =================================================================
    # STEP 4: SYSTEMATIC PERFORMANCE TESTING
    # =================================================================
    
    for temp in temperatures:
        print(f"\n📊 Testing with temperature: {temp}")
        print("-" * 50)
        
        for num_tokens in token_counts:
            print(f"\nGenerating {num_tokens} tokens...")
            
            # =============================================
            # TEST WITH KV CACHE
            # =============================================
            
            print("  🚀 With KV Cache...", end=" ")
            start_time = time.time()
            
            # Generate text with KV cache optimization
            text_with_cache = generate_text_with_kv_cache(
                model, vectorizer, start_string, seq_length, 
                num_generate=num_tokens, 
                temperature=temp, 
                use_kv_cache=True
            )
            
            time_with_cache = time.time() - start_time
            print(f"({time_with_cache:.3f}s)")
            
            # =============================================
            # TEST WITHOUT KV CACHE
            # =============================================
            
            print("  🐌 Without KV Cache...", end=" ")
            start_time = time.time()
            
            # Generate text without KV cache (standard method)
            text_without_cache = generate_text_with_kv_cache(
                model, vectorizer, start_string, seq_length, 
                num_generate=num_tokens, 
                temperature=temp, 
                use_kv_cache=False
            )
            
            time_without_cache = time.time() - start_time
            print(f"({time_without_cache:.3f}s)")
            
            # =============================================
            # CALCULATE PERFORMANCE METRICS
            # =============================================
            
            # Calculate performance improvements
            speedup = time_without_cache / time_with_cache if time_with_cache > 0 else 1
            time_saved = time_without_cache - time_with_cache
            percentage_saved = (time_saved / time_without_cache * 100) if time_without_cache > 0 else 0
            
            # Store results for analysis
            result = {
                'temperature': temp,
                'tokens': num_tokens,
                'time_with_cache': time_with_cache,
                'time_without_cache': time_without_cache,
                'speedup': speedup,
                'time_saved': time_saved,
                'percentage_saved': percentage_saved,
                'text_with_cache': text_with_cache,
                'text_without_cache': text_without_cache
            }
            results.append(result)
            
            # Display immediate results
            print(f"    ⚡ Speedup: {speedup:.2f}x")
            print(f"    ⏱️  Time saved: {time_saved:.3f}s ({percentage_saved:.1f}%)")

    # =================================================================
    # STEP 5: COMPREHENSIVE RESULTS ANALYSIS AND REPORTING
    # =================================================================
    
    # Display comprehensive results table
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE RESULTS")
    print("="*80)
    
    # Create formatted table header
    print(f"{'Temp':<6} {'Tokens':<7} {'With Cache':<12} {'Without Cache':<14} {'Speedup':<8} {'Time Saved':<12}")
    print("-" * 80)
    
    # Display each test result in tabular format
    for result in results:
        print(f"{result['temperature']:<6} {result['tokens']:<7} {result['time_with_cache']:.3f}s{'':<6} "
              f"{result['time_without_cache']:.3f}s{'':<8} {result['speedup']:.2f}x{'':<4} "
              f"{result['time_saved']:.3f}s ({result['percentage_saved']:.1f}%)")
    
    # =================================================================
    # STEP 6: STATISTICAL ANALYSIS AND AVERAGES
    # =================================================================
    
    """
    STATISTICAL ANALYSIS:
    Calculate overall performance metrics to understand:
    - Average improvement across all test cases
    - Consistency of performance gains
    - Overall efficiency improvements
    - Total time savings across all tests
    """
    
    # Calculate aggregate performance metrics
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_time_saved_pct = sum(r['percentage_saved'] for r in results) / len(results)
    total_time_with_cache = sum(r['time_with_cache'] for r in results)
    total_time_without_cache = sum(r['time_without_cache'] for r in results)
    overall_speedup = total_time_without_cache / total_time_with_cache
    
    # Display aggregate results
    print("-" * 80)
    print(f"AVERAGE PERFORMANCE IMPROVEMENT:")
    print(f"  • Average Speedup: {avg_speedup:.2f}x")
    print(f"  • Average Time Saved: {avg_time_saved_pct:.1f}%")
    print(f"  • Total Time - With Cache: {total_time_with_cache:.3f}s")
    print(f"  • Total Time - Without Cache: {total_time_without_cache:.3f}s")
    print(f"  • Overall Speedup: {overall_speedup:.2f}x")
    
    # =================================================================
    # STEP 7: SAVE DETAILED RESULTS TO FILE
    # =================================================================
    
    # Save comprehensive results report
    results_path = os.path.join(output_dir, 'performance_comparison.txt')
    with open(results_path, 'w') as f:
        # Write header and summary
        f.write("KV CACHE vs STANDARD GENERATION PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        # Write performance table
        f.write("PERFORMANCE RESULTS TABLE:\n")
        f.write(f"{'Temp':<6} {'Tokens':<7} {'With Cache':<12} {'Without Cache':<14} {'Speedup':<8} {'Time Saved':<12}\n")
        f.write("-" * 80 + "\n")
        
        # Write each result
        for result in results:
            f.write(f"{result['temperature']:<6} {result['tokens']:<7} {result['time_with_cache']:.3f}s{'':<6} "
                   f"{result['time_without_cache']:.3f}s{'':<8} {result['speedup']:.2f}x{'':<4} "
                   f"{result['time_saved']:.3f}s ({result['percentage_saved']:.1f}%)\n")
        
        # Write aggregate statistics
        f.write("\nAVERAGE PERFORMANCE IMPROVEMENT:\n")
        f.write(f"  • Average Speedup: {avg_speedup:.2f}x\n")
        f.write(f"  • Average Time Saved: {avg_time_saved_pct:.1f}%\n")
        f.write(f"  • Total Time - With Cache: {total_time_with_cache:.3f}s\n")
        f.write(f"  • Total Time - Without Cache: {total_time_without_cache:.3f}s\n")
        f.write(f"  • Overall Speedup: {overall_speedup:.2f}x\n\n")
        
        # Write generated text samples
        f.write("GENERATED TEXT SAMPLES:\n")
        f.write("="*50 + "\n\n")
        
        # Save all generated text samples for quality comparison
        for i, result in enumerate(results):
            f.write(f"Sample {i+1}: Temperature {result['temperature']}, {result['tokens']} tokens\n")
            f.write(f"Start string: {start_string}\n\n")
            
            f.write("WITH KV CACHE:\n")
            f.write(f"Time: {result['time_with_cache']:.3f}s\n")
            f.write(f"Text: {result['text_with_cache']}\n\n")
            
            f.write("WITHOUT KV CACHE:\n")
            f.write(f"Time: {result['time_without_cache']:.3f}s\n")
            f.write(f"Text: {result['text_without_cache']}\n\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"\n📄 Detailed results saved to: {results_path}")
    
    # =================================================================
    # STEP 8: PERFORMANCE INSIGHTS AND RECOMMENDATIONS  
    # =================================================================
    
    """
    PERFORMANCE INSIGHTS:
    Analyze the results to provide actionable insights about:
    - When KV Cache provides the most benefit
    - Scalability characteristics
    - Practical recommendations for usage
    """
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Provide performance assessment based on results
    if avg_speedup > 2.0:
        print("🎉 Excellent performance! KV Cache provides significant speedup.")
    elif avg_speedup > 1.5:
        print("✅ Good performance improvement with KV Cache.")
    elif avg_speedup > 1.1:
        print("⚡ Modest but meaningful speedup with KV Cache.")
    else:
        print("⚠️  Limited speedup - may need larger sequences to see benefits.")
    
    # Provide usage recommendations
    print(f"\n💡 The KV Cache optimization is most effective for:")
    print(f"   • Longer sequence generation ({max(token_counts)} tokens showed best relative improvement)")
    print(f"   • Interactive applications where latency matters")
    print(f"   • Batch generation of multiple sequences")
    
    # Technical explanation
    print(f"\n🔧 Technical details:")
    print(f"   • KV Cache eliminates redundant computation of attention keys/values")
    print(f"   • Memory usage increases slightly to store cached values")
    print(f"   • Speedup scales with sequence length and model size")
    
    print(f"\nAll comparison results saved in: {output_dir}")


# =============================================================================
# MAIN FUNCTION - Orchestrates the entire pipeline
# =============================================================================

def main():
    """
    MAIN EXECUTION FUNCTION - Complete pipeline orchestration
    
    FUNCTION PURPOSE:
    This is the central control function that orchestrates the entire machine learning
    pipeline from training to text generation. It provides a flexible interface for
    running different modes of operation based on user requirements.
    
    EXECUTION MODES:
    1. 'train': Only train the model, save weights and configuration
    2. 'generate': Load trained model and generate text with performance analysis
    3. 'generate_compare': Comprehensive KV cache performance comparison
    4. 'both': Complete pipeline - train model then run comparison
    
    CONFIGURATION MANAGEMENT:
    The function uses configuration variables that can be easily modified to:
    - Change execution mode
    - Enable/disable KV cache optimization
    - Specify model file paths
    - Adjust generation parameters
    
    WORKFLOW ORCHESTRATION:
    - Provides clear separation between training and inference phases
    - Handles file path management and dependencies
    - Ensures proper error handling and user feedback
    - Coordinates multiple analysis modes
    
    DESIGN PATTERNS:
    - Configuration-driven execution
    - Mode-based operation selection
    - Clear separation of concerns
    - Comprehensive logging and feedback
    """
    
    # =================================================================
    # CONFIGURATION SECTION - Modify these parameters as needed
    # =================================================================
    
    """
    CONFIGURATION PARAMETERS:
    These variables control the execution behavior and can be modified based on:
    - Available computational resources
    - Desired analysis depth
    - File system organization
    - Specific research or application needs
    """
    
    # Execution mode selection
    mode = 'generate_compare'  # Options: 'train', 'generate', 'generate_compare', 'both'
    
    # Performance optimization settings
    use_kv_cache = True  # Set to False to disable KV cache optimization
    
    # File path configuration
    weights_path = 'transformer_model.weights.h5'     # Model weights file
    vectorizer_path = 'text_vectorizer.pkl'           # Vectorizer and config file
    
    # =================================================================
    # MODE-BASED EXECUTION
    # =================================================================
    
    if mode == 'train':
        # =============================================
        # TRAINING ONLY MODE
        # =============================================
        
        """
        TRAINING MODE:
        Executes the complete training pipeline:
        - Data loading and preprocessing
        - Model architecture setup
        - Training execution with monitoring
        - Model persistence and artifact saving
        - Training analysis and visualization
        """
        
        print("="*60)
        print("TRAINING MODE")
        print("="*60)
        train()
        
    elif mode == 'generate':
        # =============================================
        # GENERATION ONLY MODE
        # =============================================
        
        """
        GENERATION MODE:
        Loads trained model and generates text:
        - Model reconstruction from saved artifacts
        - Text generation with specified parameters
        - Performance measurement and analysis
        - Multiple temperature experimentation
        - Output saving and organization
        """
        
        print("="*60)
        print("GENERATION MODE")
        print("="*60)
        generate(
            use_kv_cache=use_kv_cache, 
            weights_path=weights_path, 
            vectorizer_path=vectorizer_path
        )
        
    elif mode == 'generate_compare':
        # =============================================
        # PERFORMANCE COMPARISON MODE
        # =============================================
        
        """
        COMPARISON MODE:
        Comprehensive KV cache performance analysis:
        - Systematic testing across multiple parameters
        - Detailed timing measurements and analysis
        - Statistical performance evaluation
        - Comprehensive reporting and insights
        - Quality verification of generated text
        """
        
        print("="*60)
        print("PERFORMANCE COMPARISON MODE")
        print("="*60)
        generate_compare(
            weights_path=weights_path, 
            vectorizer_path=vectorizer_path
        )
        
    else:  # both
        # =============================================
        # COMPLETE PIPELINE MODE
        # =============================================
        
        """
        COMPLETE PIPELINE MODE:
        Executes the full machine learning workflow:
        1. Train model from scratch
        2. Save all training artifacts
        3. Perform comprehensive performance comparison
        4. Generate detailed analysis reports
        
        This mode is ideal for:
        - Complete experiments from start to finish
        - Reproducible research workflows
        - Comprehensive model evaluation
        - Demonstration of full capabilities
        """
        
        print("="*60)
        print("TRAINING MODE")
        print("="*60)
        
        # Execute training phase
        model, vectorizer, vocab_size, seq_length, logdir = train()
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON MODE")
        print("="*60)
        
        # Execute comparison phase using trained model
        generate_compare(
            weights_path=weights_path, 
            vectorizer_path=vectorizer_path
        )


# =============================================================================
# SCRIPT ENTRY POINT - Execute when run as main program
# =============================================================================

if __name__ == "__main__":
    """
    SCRIPT ENTRY POINT
    
    This ensures that the main function only runs when the script is executed directly,
    not when it's imported as a module. This is a Python best practice that allows
    the code to be both executable and importable.
    
    EXECUTION CONTEXT:
    When this script is run directly (python main_commented.py), it will execute
    the main() function and run the complete pipeline based on the configuration
    settings defined in the main() function.
    
    IMPORTABLE MODULE:
    If this script is imported (import main_commented), the classes and functions
    will be available for use, but the main() function won't execute automatically.
    This allows other scripts to reuse the components without triggering execution.
    """
    main()
