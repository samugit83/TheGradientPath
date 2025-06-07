#!/usr/bin/env python3
"""
Complete script to implement Transformers for text generation using the Shakespeare dataset.
Build, train, and evaluate Transformer models for text generation using TensorFlow and Keras.
This script demonstrates how to use the Transformer architecture for natural language processing
tasks, specifically text generation based on learning patterns from Shakespeare's works.
"""
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, LayerNormalization, Dropout, Embedding, 
    MultiHeadAttention, TextVectorization
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import visualkeras

# Configure TensorFlow to use GPU if available
# This section checks if a GPU is available and configures TensorFlow to use it.
# Using a GPU can significantly speed up model training, especially for large models.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once.
        # This is useful when sharing the GPU with other processes or when you want to
        # run multiple models simultaneously.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured.")
    except RuntimeError as e:
        # Print an error message if GPU setup fails.
        print(f"Error setting up GPU memory growth: {e}")
else:
    # If no GPU is found, TensorFlow will use the CPU.
    print("No GPU detected, using CPU.")


class TransformerBlock(Layer):
    """
    Implements a single Transformer Block for text generation.
    A Transformer Block is the fundamental building unit of the Transformer architecture.
    It consists of:
    1. Multi-Head Self-Attention: Allows each token to attend to all other tokens in the sequence
    2. Feed-Forward Network (FFN): Position-wise neural network for feature transformation
    3. Layer Normalization: Applied before each sub-layer for training stability
    4. Dropout: Applied for regularization to prevent overfitting
    5. Residual Connections: Adds input to output (input + SubLayer(input)) for gradient flow
    
    In text generation, this block helps the model understand:
    - Which words are related to each other in the sequence
    - How to transform word representations based on context
    - Long-range dependencies between words (e.g., subject-verb agreement)
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initializes the TransformerBlock for text processing.
        Args:
            embed_dim (int): Dimensionality of the token embeddings and all internal representations.
                            This is the "width" of the model - how many features each token has.
            num_heads (int): Number of attention heads. Multiple heads allow the model to attend
                           to different types of relationships simultaneously (e.g., syntactic vs semantic).
            ff_dim (int): Dimensionality of the inner layer of the Feed-Forward Network.
                         Typically 4x embed_dim to provide more representational capacity.
            rate (float): Dropout rate for regularization (fraction of neurons to randomly disable).
        """
        super(TransformerBlock, self).__init__()
        
        # Multi-Head Self-Attention layer using TensorFlow's built-in implementation
        # This is more efficient and stable than custom implementations
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        
        # Position-wise Feed-Forward Network (FFN)
        # Applied independently to each position in the sequence
        # Architecture: Dense(ff_dim, ReLU) -> Dense(embed_dim, Linear)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),  # Expands to ff_dim for more capacity
            Dense(embed_dim),                 # Projects back to embed_dim
        ])
        
        # Layer Normalization layers for training stability
        # These normalize activations across the embedding dimension
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # After attention
        self.layernorm2 = LayerNormalization(epsilon=1e-6)  # After FFN
        
        # Dropout layers for regularization
        self.dropout1 = Dropout(rate)  # Applied to attention output
        self.dropout2 = Dropout(rate)  # Applied to FFN output

    def call(self, inputs, training=False):
        """
        Forward pass of the TransformerBlock for text processing.
        Args:
            inputs: Input token embeddings (batch_size, sequence_length, embed_dim).
                   Each token in the sequence is represented by an embed_dim-dimensional vector.
            training (bool): Whether the model is in training mode (affects dropout).
        Returns:
            Output tensor with same shape as input but with contextualized representations.
        """
        # 1. MULTI-HEAD SELF-ATTENTION SUB-LAYER
        # =======================================
        # CONCEPTUAL PURPOSE: Each token "looks at" all other tokens in the sequence
        # to understand its meaning in context. This is crucial for text generation
        # because the meaning of a word depends heavily on surrounding words.
        #
        # EXAMPLE: In "The bank of the river", the word "bank" should attend strongly
        # to "river" to understand it means riverbank, not financial institution.
        #
        # HOW IT WORKS FOR TEXT:
        # - Each token creates Query (what am I looking for?), Key (what do I offer?), 
        #   and Value (what information do I carry?) representations
        # - Attention scores are computed between all token pairs
        # - Each token's output is a weighted sum of all tokens' Values
        # - Multiple heads capture different types of relationships (syntax, semantics, etc.)
        #
        # INPUT/OUTPUT SHAPES:
        # inputs shape: (batch_size, sequence_length, embed_dim)
        # For text: sequence_length = number of tokens, embed_dim = token representation size
        attn_output = self.att(inputs, inputs)  # Self-attention: query=key=value=inputs
        
        # DROPOUT ON ATTENTION OUTPUT:
        # Randomly zeros out some attention features during training
        # Prevents overfitting to specific attention patterns
        attn_output = self.dropout1(attn_output, training=training)
        
        # FIRST RESIDUAL CONNECTION + LAYER NORMALIZATION:
        # ================================================
        # RESIDUAL CONNECTION (inputs + attn_output):
        # - Allows information to flow directly through the network
        # - Enables training of very deep networks without vanishing gradients
        # - Lets the model learn identity mapping if attention isn't helpful
        #
        # LAYER NORMALIZATION:
        # - Normalizes each token's representation independently
        # - For each token position, normalizes across the embed_dim features
        # - Helps with training stability and convergence speed
        out1 = self.layernorm1(inputs + attn_output)

        # 2. FEED-FORWARD NETWORK (FFN) SUB-LAYER
        # ========================================
        # CONCEPTUAL PURPOSE: While attention captures relationships between tokens,
        # the FFN processes each token's representation independently with non-linear
        # transformations. This adds computational capacity for complex feature learning.
        #
        # WHY THIS MATTERS FOR TEXT: After attention has gathered contextual information,
        # the FFN can transform this information into more useful representations.
        # For example, it might learn to recognize that certain attention patterns
        # indicate specific grammatical structures or semantic concepts.
        #
        # ARCHITECTURE DETAILS:
        # - First Dense layer: embed_dim -> ff_dim (expansion for more capacity)
        # - ReLU activation: Adds non-linearity for complex transformations
        # - Second Dense layer: ff_dim -> embed_dim (compression back to original size)
        # - Applied position-wise: same transformation for each token independently
        ffn_output = self.ffn(out1)
        
        # DROPOUT ON FFN OUTPUT:
        # Regularizes the FFN transformations to prevent overfitting
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # SECOND RESIDUAL CONNECTION + LAYER NORMALIZATION:
        # =================================================
        # Same benefits as the first residual connection and layer norm
        # Ensures stable gradients and allows the model to preserve information
        # if the FFN transformation isn't beneficial for a particular token
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(Model):
    """
    Complete Transformer model for text generation.
    This model implements the decoder-style Transformer architecture suitable for
    autoregressive text generation (predicting the next token given previous tokens).
    
    Architecture components:
    1. Token Embedding: Converts token IDs to dense vectors
    2. Positional Encoding: Adds position information to embeddings
    3. Transformer Blocks: Stack of attention and feed-forward layers
    4. Output Projection: Maps final representations to vocabulary probabilities
    
    The model is trained to predict the next token in a sequence, learning patterns
    and structures from the training text (Shakespeare in this case).
    """
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        """
        Initializes the complete Transformer model for text generation.
        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            embed_dim (int): Dimensionality of token embeddings and all representations.
            num_heads (int): Number of attention heads in each Transformer block.
            ff_dim (int): Dimensionality of feed-forward network inner layer.
            num_layers (int): Number of Transformer blocks to stack.
            seq_length (int): Maximum sequence length for positional encoding.
        """
        super(TransformerModel, self).__init__()
        
        # TOKEN EMBEDDING LAYER:
        # ======================
        # The Embedding layer transforms each integer token ID in the input sequence
        # into a dense vector of fixed size (embed_dim). This is done by maintaining
        # a trainable lookup table (a weight matrix) of shape (vocab_size, embed_dim),
        # where each row corresponds to the embedding vector for a specific token in the vocabulary.
        #
        # When the model receives an input of shape (batch_size, sequence_length) containing
        # integer token IDs, the Embedding layer replaces each token ID with its corresponding
        # embedding vector from the lookup table. The result is a 3D tensor of shape:
        #   (batch_size, sequence_length, embed_dim)
        # - batch_size: number of sequences in the batch
        # - sequence_length: number of tokens in each input sequence
        # - embed_dim: dimensionality of each token's embedding vector
        #
        # For example, if batch_size=32, sequence_length=50, and embed_dim=256,
        # the output shape will be (32, 50, 256).
        #
        # These embedding vectors are initialized randomly and are learned/updated
        # during training via backpropagation, allowing the model to capture semantic
        # and syntactic relationships between tokens.
        self.embedding = Embedding(vocab_size, embed_dim)
        
        # Positional Encoding
        # Since attention has no inherent notion of position, we add positional
        # information to help the model understand word order.
        # The resulting self.pos_encoding has shape (1, seq_length, embed_dim):
        #   - 1: batch dimension (so it can be broadcasted to any batch size)
        #   - seq_length: the maximum sequence length supported by the model
        #   - embed_dim: the dimensionality of each token embedding
        # This shape allows us to add positional encodings directly to the embedded input tokens.
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)
        
        # Stack of Transformer Blocks
        # Each block processes the sequence, building increasingly complex representations
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) 
            for _ in range(num_layers)
        ]
        
        # Output projection layer
        # This is a linear (no activation) Dense layer.
        # It maps the final embed_dim representations to logits for each vocabulary token.
        # The softmax for probabilities is applied later (e.g., in loss or sampling).
        self.dense = Dense(vocab_size)  # Linear output (no activation)

    def positional_encoding(self, seq_length, embed_dim):
        """
        Creates sinusoidal positional encodings for the input sequence.
        Positional encoding helps the model understand the order of tokens since
        the attention mechanism itself is position-agnostic.
        
        The encoding uses sine and cosine functions of different frequencies:
        - Even dimensions use sine functions
        - Odd dimensions use cosine functions
        - Different positions get unique encoding patterns
        - Similar positions get similar encodings
        
        Args:
            seq_length (int): Maximum sequence length to encode.
            embed_dim (int): Dimensionality of the embeddings.
        Returns:
            Positional encoding tensor of shape (1, seq_length, embed_dim).
        """
        # Create position and dimension indices
        # pos: [0, 1, 2, ..., seq_length-1] for each position in sequence
        # i: [0, 1, 2, ..., embed_dim-1] for each embedding dimension
        angle_rads = self.get_angles(
            np.arange(seq_length)[:, np.newaxis],    # Shape: (seq_length, 1)
            np.arange(embed_dim)[np.newaxis, :],     # Shape: (1, embed_dim)
            embed_dim
        )
        
        # Apply sine to even indices (0, 2, 4, ...)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        # Add batch dimension and convert to TensorFlow tensor
        pos_encoding = angle_rads[np.newaxis, ...]  # Shape: (1, seq_length, embed_dim)
        # This function returns the positional encoding tensor of shape (1, seq_length, embed_dim)
        # with dtype float32, ready to be added to the input embeddings.
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, embed_dim):
        """
        Calculates the angles for positional encoding.
        Uses the formula from "Attention Is All You Need":
        angle(pos, 2i) = pos / 10000^(2i/embed_dim)
        angle(pos, 2i+1) = pos / 10000^(2i/embed_dim)
        
        Args:
            pos: Position indices.
            i: Dimension indices.
            embed_dim: Embedding dimensionality.
        Returns:
            Angle values for sine/cosine computation.
        """
        # Calculate the angle rates using the formula from the paper
        # Higher dimensions get lower frequencies (slower oscillations)
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def call(self, inputs, training=False):
        """
        Forward pass of the complete Transformer model.
        Args:
            inputs: Token IDs tensor (batch_size, sequence_length).
                   Each element is an integer representing a token in the vocabulary.
            training (bool): Whether the model is in training mode.
        Returns:
            Logits tensor (batch_size, sequence_length, vocab_size).
            For each position, provides unnormalized probabilities over the vocabulary.
        """
        # Get the actual sequence length (may be less than max seq_length)
        seq_len = tf.shape(inputs)[1]
        
        # 1. CONVERT TOKEN IDS TO EMBEDDINGS
        # ==================================
        # Transform integer token IDs into dense vector representations
        # Each token ID (0 to vocab_size-1) maps to a learned embed_dim vector
        # These embeddings capture semantic relationships between tokens
        x = self.embedding(inputs)  # Shape: (batch_size, seq_len, embed_dim)
        
        # 2. ADD POSITIONAL ENCODING
        # ==========================
        # The shape of self.pos_encoding is (1, seq_length, embed_dim):
        #   - 1: batch dimension (for broadcasting)
        #   - seq_length: maximum sequence length used during model initialization
        #   - embed_dim: embedding dimension
        # We slice the positional encoding to match the actual sequence length of the input.
        # This ensures that the positional information aligns with the input embeddings.
        x += self.pos_encoding[:, :seq_len, :]  # Broadcasting addition; pos_encoding shape: (1, seq_len, embed_dim)
        
        # 3. PASS THROUGH TRANSFORMER BLOCKS
        # ===================================
        # Each block applies self-attention and feed-forward transformations
        # The sequence representations become increasingly contextualized
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # 4. PROJECT TO VOCABULARY SIZE
        # =============================
        # Convert the final representations to logits over the vocabulary
        # Each position gets a probability distribution over all possible next tokens
        output = self.dense(x)  # Shape: (batch_size, seq_len, vocab_size)
        return output


def create_sequences(text, seq_length):
    """
    Generate input and target sequences for training the text generation model.
    Converts a long text into overlapping sequences suitable for next-token prediction.
    
    For text generation, we use a sliding window approach:
    - Input sequence: tokens [i, i+1, ..., i+seq_length-1]
    - Target sequence: tokens [i+1, i+2, ..., i+seq_length]
    
    This creates a "teacher forcing" setup where the model learns to predict
    each token given all previous tokens in the sequence.
    
    Example with seq_length=3:
    Text: "To be or not to be"
    Tokenized: [1, 2, 3, 4, 1, 2]
    
    Input sequences:  [[1, 2, 3], [2, 3, 4], [3, 4, 1]]
    Target sequences: [[2, 3, 4], [3, 4, 1], [4, 1, 2]]
    
    Args:
        text (np.array): Tokenized text as integer array.
        seq_length (int): Length of each training sequence.
    Returns:
        input_seqs (np.array): Input sequences for training.
        target_seqs (np.array): Target sequences for training.
    """
    input_seqs = []
    target_seqs = []
    
    # Create overlapping sequences with sliding window
    for i in range(len(text) - seq_length):
        # Input: current position to current position + seq_length
        input_seq = text[i:i + seq_length]
        
        # Target: shifted by one position (next token prediction)
        target_seq = text[i + 1:i + seq_length + 1]
        
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    
    return np.array(input_seqs), np.array(target_seqs)


def generate_text(model, vectorizer, start_string, seq_length, num_generate=100, temperature=1.0):
    """
    Generate text using the trained Transformer model.
    This function implements autoregressive text generation, where each new token
    is predicted based on all previously generated tokens.
    
    The generation process:
    1. Convert start string to token IDs
    2. For each new token to generate:
       a. Pass current sequence through the model
       b. Get probability distribution over vocabulary for next token
       c. Sample next token using temperature-controlled sampling
       d. Add sampled token to sequence
       e. Repeat until desired length is reached
    
    Args:
        model: Trained Transformer model.
        vectorizer: TextVectorization layer used during training.
        start_string (str): Initial text to start generation from.
        seq_length (int): Maximum sequence length the model was trained on.
        num_generate (int): Number of tokens to generate.
        temperature (float): Controls randomness in sampling:
                           - Low (0.1-0.7): More focused, deterministic text
                           - High (1.0-2.0): More random, creative text
    Returns:
        Generated text string.
    """
    # ========================================================================
    # Let's dissect this function with surgical precision to understand the 
    # deep mechanics of autoregressive text generation:
    # ========================================================================
    
    # ========================================================================
    # PHASE 1: INPUT PREPROCESSING AND TENSOR SHAPE MANAGEMENT
    # ========================================================================
    
    # The first critical step involves converting our human-readable text into 
    # the numerical format our model expects. The `vectorizer([start_string]).numpy()` 
    # call performs several operations under the hood:
    # - Text tokenization: splits text into discrete tokens (words or subwords)
    # - Vocabulary lookup: maps each token to its corresponding integer ID
    # - Tensor creation: wraps the IDs in a NumPy array with shape `(1, sequence_length)`
    input_eval = vectorizer([start_string]).numpy()
    
    # The shape management logic is crucial for maintaining consistent input dimensions. 
    # Our model expects exactly `seq_length` tokens, so we implement intelligent 
    # padding and truncation:
    
    if input_eval.shape[1] < seq_length:
        # **Padding strategy**: When input is shorter than `seq_length`, we prepend zeros. 
        # This creates a left-padded sequence where the meaningful tokens appear at the end, 
        # allowing the model to focus on the most recent context for prediction.
        padding = np.zeros((1, seq_length - input_eval.shape[1]))
        input_eval = np.concatenate((padding, input_eval), axis=1)
    elif input_eval.shape[1] > seq_length:
        # **Truncation strategy**: When input exceeds `seq_length`, we take the rightmost 
        # tokens `[:, -seq_length:]`. This preserves the most recent context, which is 
        # typically most relevant for next-token prediction.
        input_eval = input_eval[:, -seq_length:]

    # Convert to TensorFlow tensor for model input
    input_eval = tf.convert_to_tensor(input_eval)
    
    # Initialize storage for generated tokens
    text_generated = []  # Store generated tokens

    # ========================================================================
    # PHASE 2: THE AUTOREGRESSIVE GENERATION LOOP
    # ========================================================================
    
    # The heart of text generation lies in the autoregressive loop - a process that 
    # mirrors how humans generate language by considering previous words to predict 
    # the next one. Each iteration follows a precise sequence:
    
    for i in range(num_generate):
        
        # ====================================================================
        # STEP 2.1: FORWARD PASS THROUGH THE TRANSFORMER
        # ====================================================================
        
        # This single line triggers a complex cascade of operations:
        # - **Embedding lookup**: Each token ID is converted to a dense `embed_dim`-dimensional vector
        # - **Positional encoding addition**: Mathematical sine/cosine patterns are added to embed positional information
        # - **Multi-layer attention processing**: Each transformer block applies self-attention mechanisms, 
        #   allowing tokens to attend to relevant parts of the sequence
        # - **Feed-forward processing**: Dense layers transform the attended representations
        # - **Final projection**: The last Dense layer projects to vocabulary size, producing raw logits 
        #   for each possible next token
        predictions = model(input_eval)  # Shape: (batch_size, seq_length, vocab_size)

        # ====================================================================
        # STEP 2.2: LOGIT EXTRACTION AND CAUSAL MASKING
        # ====================================================================
        
        # This line implements causal masking by selecting only the predictions for the last position. 
        # In our model architecture, each position predicts the next token, so `predictions[0, -1, :]` 
        # contains the probability distribution over the entire vocabulary for what should come after 
        # our current sequence. The indexing `[0, -1, :]` means:
        # - `0`: First (and only) batch element
        # - `-1`: Last sequence position (where we want to predict the next token)
        # - `:`: All vocabulary dimensions
        predictions = predictions[0, -1, :]  # Shape: (vocab_size,)

        # ====================================================================
        # STEP 2.3: TEMPERATURE SCALING AND PROBABILITY DISTRIBUTION SHAPING
        # ====================================================================
        
        # Temperature scaling is a crucial technique for controlling the "creativity" vs 
        # "conservatism" trade-off in generation:
        # - **Mathematical foundation**: We're scaling the logits before applying softmax. 
        #   Since softmax is `exp(x_i) / sum(exp(x_j))`, dividing logits by temperature τ 
        #   gives us `exp(x_i/τ) / sum(exp(x_j/τ))`
        # - **Temperature < 1 (e.g., 0.7)**: Makes the distribution more peaked, increasing 
        #   the probability of the most likely tokens. This produces more conservative, predictable text.
        # - **Temperature > 1**: Flattens the distribution, giving lower-probability tokens 
        #   more chance to be selected. This increases creativity but risks incoherence.
        # - **Temperature = 1**: No modification - uses the model's raw learned distribution.
        predictions = predictions / temperature
        
        # ====================================================================
        # STEP 2.4: STOCHASTIC SAMPLING WITH CATEGORICAL DISTRIBUTION
        # ====================================================================
        
        # Multinomial sampling is a probabilistic method for selecting the next token in text generation.
        # Instead of always choosing the token with the highest probability (which would be deterministic and repetitive),
        # multinomial sampling draws a token according to the full probability distribution predicted by the model.
        #
        # Here's how it works in detail:
        # 1. The model outputs a vector of logits (unnormalized scores) for each possible token in the vocabulary.
        # 2. These logits are optionally scaled by a "temperature" parameter to control randomness.
        # 3. The logits are then converted into probabilities using the softmax function.
        # 4. Multinomial sampling treats these probabilities as defining a categorical distribution:
        #    - Each token has a chance of being selected proportional to its probability.
        #    - This is like rolling a weighted die, where more likely tokens have a higher chance but less likely tokens can still be chosen.
        # 5. In TensorFlow, `tf.random.categorical` performs this sampling:
        #    - It expects a 2D tensor of logits (batch_size, vocab_size), so we use `tf.expand_dims(predictions, 0)` to add a batch dimension.
        #    - `num_samples=1` means we draw one sample (one token) from the distribution.
        #    - The result is a tensor of shape (1, 1), so we extract the scalar value with `[0, 0]`.
        #    - `.numpy()` converts the result from a TensorFlow tensor to a standard Python integer.
        #
        # This approach introduces controlled randomness, allowing the model to generate more diverse and creative text,
        # while still being guided by the learned probabilities. It avoids the pitfall of always picking the most likely token,
        # which can lead to dull and repetitive outputs.
        predicted_id = tf.random.categorical(
            tf.expand_dims(predictions, 0),
            num_samples=1
        )[0, 0].numpy()

        # ====================================================================
        # STEP 2.5: CONTEXT WINDOW MANAGEMENT AND SLIDING WINDOW TECHNIQUE
        # ====================================================================
        
        # This implements a sliding window approach essential for maintaining computational efficiency:
        # - **Token appending**: The newly generated token is added to our sequence
        # - **Window sliding**: We keep only the most recent `seq_length` tokens, discarding older context
        # - **Memory efficiency**: This prevents sequences from growing indefinitely, keeping memory 
        #   usage constant regardless of generation length
        # - **Attention relevance**: Recent tokens typically have more influence on next-token 
        #   prediction than distant ones
        input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)  # Append new token
        input_eval = input_eval[:, -seq_length:]                              # Maintain fixed window size
        input_eval = tf.convert_to_tensor(input_eval)                         # Convert back to tensor

        # ====================================================================
        # STEP 2.6: VOCABULARY LOOKUP AND TOKEN-TO-TEXT CONVERSION
        # ====================================================================
        
        # The final step converts numerical predictions back to human-readable text:
        # - **Vocabulary retrieval**: `get_vocabulary()` returns the learned word-to-ID mapping
        # - **Bounds checking**: The `if predicted_id < len(vocab)` prevents index errors 
        #   if the model predicts invalid token IDs
        # - **Reverse lookup**: `vocab[predicted_id]` retrieves the actual word/token 
        #   corresponding to the predicted ID
        vocab = vectorizer.get_vocabulary()
        if predicted_id < len(vocab):
            text_generated.append(vocab[predicted_id])

    # ========================================================================
    # ADVANCED CONSIDERATIONS AND EDGE CASES
    # ========================================================================
    
    # Several subtle but important details make this implementation robust:
    #
    # 1. **Tensor type conversions**: Notice the frequent `.numpy()` and `tf.convert_to_tensor()` 
    #    calls. This handles TensorFlow's eager execution mode and ensures compatibility between 
    #    NumPy operations and TensorFlow operations.
    #
    # 2. **Memory management**: The sliding window approach prevents memory explosion during 
    #    long generation sequences, crucial for production deployment.
    #
    # 3. **Padding token handling**: Zero-padding at the beginning ensures that shorter input 
    #    sequences don't confuse the positional encoding, as the model learns that position 0 
    #    typically contains padding.
    #
    # 4. **Probabilistic vs deterministic generation**: The stochastic sampling approach mimics 
    #    human language generation better than deterministic approaches, as human language 
    #    contains natural variation and creativity.
    #
    # This generation process embodies the same principles used in GPT, ChatGPT, and other 
    # state-of-the-art language models - the difference lies primarily in scale (billions vs 
    # millions of parameters) and training data diversity. The fundamental autoregressive 
    # generation mechanism remains conceptually identical across all modern language models.
    
    # ========================================================================
    # FINAL TEXT ASSEMBLY AND RETURN
    # ========================================================================
    # Join the start string with all generated tokens
    return start_string + ' ' + ' '.join(text_generated)


def main():
    """
    Main function implementing the complete text generation pipeline.
    This function demonstrates end-to-end Transformer-based text generation:
    1. Data loading and preprocessing
    2. Model architecture definition
    3. Training with monitoring
    4. Text generation and evaluation
    5. Results visualization and saving
    """
    
    # ========================================================================
    # STEP 1: LOAD AND EXPLORE THE SHAKESPEARE DATASET
    # ========================================================================
    print("Loading Shakespeare dataset...")
    
    # Download the Shakespeare text dataset from TensorFlow's data repository
    # This dataset contains the complete works of Shakespeare, providing rich
    # examples of English language patterns, vocabulary, and literary style
    path_to_file = get_file(
        'shakespeare.txt', 
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    )
    
    # Read the text file and decode it as UTF-8
    # The 'rb' mode reads in binary, then we decode to handle any special characters
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    
    # Display basic information about the dataset
    print(f"Dataset loaded. Text length: {len(text)} characters")
    print("Preview of the dataset:")
    print(text[:500])  # Show first 500 characters to understand the data structure

    # ========================================================================
    # STEP 2: TEXT PREPROCESSING AND TOKENIZATION
    # ========================================================================
    print("\nPreprocessing text data...")
    
    # Define preprocessing parameters
    vocab_size = 10000   # Maximum number of unique tokens to keep
    seq_length = 100     # Length of input sequences for training
    
    # TOKENIZATION WITH TEXTVECTORIZATION:
    # ====================================
    # The TextVectorization layer is responsible for converting raw text into sequences of integer token IDs.
    # It operates in two main phases:
    #   1. Vocabulary building (adaptation): The vectorizer scans the entire dataset to identify the most frequent tokens (words or subwords)
    #      and assigns each a unique integer index, up to the specified maximum vocabulary size.
    #   2. Tokenization and mapping: When text is passed through the vectorizer, it splits the text into tokens (e.g., words),
    #      normalizes them (such as lowercasing and removing punctuation), and replaces each token with its corresponding integer index.
    # This transformation is essential because neural networks require numerical input rather than raw text.
    #
    # After adaptation, the vectorizer "remembers" the vocabulary it has built. Each token in the vocabulary is mapped to a unique integer,
    # and any token not in the vocabulary is mapped to a special "out-of-vocabulary" (OOV) index.
    # The vectorizer also provides methods to convert from token IDs back to the original tokens using its vocabulary.
    # For example, after adaptation, you can call `vectorizer.get_vocabulary()` to see the list of all tokens and their indices.
    # When you pass a batch of text to the vectorizer, the output is a tensor of shape (batch_size, sequence_length),
    # where each entry is the integer index of a token in the vocabulary.
    vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int')
    
    # VOCABULARY ADAPTATION:
    # ======================
    # The vectorizer must "adapt" on the entire dataset to build its vocabulary and internal mappings.
    # After calling `adapt`, the vectorizer is ready to convert any text (containing known or unknown tokens)
    # into sequences of integer indices, using the vocabulary it has learned.
    text_ds = tf.data.Dataset.from_tensor_slices([text]).batch(1)
    vectorizer.adapt(text_ds)
    # At this point, the vectorizer has:
    #   - Built a vocabulary of the most frequent tokens in the dataset (up to vocab_size)
    #   - Assigned each token a unique integer index (starting from 1; 0 is usually reserved for padding)
    #   - Set up a special index for out-of-vocabulary tokens
    #   - Is ready to transform any input text into a sequence of token IDs, and to map IDs back to tokens via `get_vocabulary()`
    # CONVERT TEXT TO INTEGER SEQUENCES:
    # ==================================
    # Transform the entire text into a sequence of integer token IDs
    vectorized_text = vectorizer([text])[0]
    print(f"Vectorized text shape: {vectorized_text.shape}")
    print(f"First 10 vectorized tokens: {vectorized_text.numpy()[:10]}")
    
    # Display vocabulary information
    vocab = vectorizer.get_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample vocabulary: {vocab[:20]}")  # Show first 20 tokens

    # ========================================================================
    # STEP 3: CREATE TRAINING SEQUENCES
    # ========================================================================
    print("\nCreating training sequences...")
    
    # SEQUENCE GENERATION FOR LANGUAGE MODELING:
    # ==========================================
    # Convert the long text into overlapping sequences for training
    # Each sequence teaches the model to predict the next token given previous tokens
    X, Y = create_sequences(vectorized_text.numpy(), seq_length)
    
    # VALIDATION AND INFORMATION:
    # ===========================
    print(f"Number of sequences generated: {len(X)}")
    
    # Ensure we have valid training data
    assert X.size > 0, "Input data X is empty"
    assert Y.size > 0, "Target data Y is empty"
    
    # Convert to TensorFlow tensors for efficient training
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)
    print(f"Shape of X (inputs): {X.shape}")
    print(f"Shape of Y (targets): {Y.shape}")
    
    # Show example of input-target pair
    print(f"Example input sequence: {X[0].numpy()[:10]}...")
    print(f"Example target sequence: {Y[0].numpy()[:10]}...")

    # ========================================================================
    # STEP 4: BUILD THE TRANSFORMER MODEL
    # ========================================================================
    print("\nBuilding Transformer model...")
    
    # HYPERPARAMETER CONFIGURATION:
    # =============================
    # These parameters control the model's capacity and behavior
    embed_dim = 256      # Dimensionality of token embeddings (model width)
    num_heads = 4        # Number of attention heads (parallel attention mechanisms)
    ff_dim = 512         # Feed-forward network inner dimension (usually 2-4x embed_dim)
    num_layers = 4       # Number of Transformer blocks (model depth)
    
    # MODEL INSTANTIATION:
    # ===================
    # Create the complete Transformer model with specified architecture
    model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        seq_length=seq_length
    )

    # MODEL INITIALIZATION:
    # ====================
    # Build the model by passing a dummy input to establish all layer shapes
    # This is necessary before compilation and training
    dummy_input = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)
    _ = model(dummy_input)

    # MODEL COMPILATION:
    # =================
    # Configure the model for training with appropriate loss and optimizer
    model.compile(
        optimizer='adam',                           # Adaptive learning rate optimizer
        loss='sparse_categorical_crossentropy'      # Appropriate for integer targets
    )
    
    # Display model architecture
    model.summary()
    print(f"Total trainable parameters: {model.count_params():,}")

    # ========================================================================
    # STEP 4.5: GENERATE MODEL ARCHITECTURE VISUALIZATION
    # ========================================================================
    print("\nGenerating model architecture visualization...")
    
    # VISUALKERAS MODEL VISUALIZATION:
    # ================================
    # VisualKeras creates beautiful, publication-ready visualizations of neural network architectures.
    # This helps in understanding the model structure, layer connections, and overall complexity.
    # The visualization shows each layer as a colored block with dimensions and connections.
    #
    # VISUALIZATION PARAMETERS:
    # - legend=True: Shows layer names and types for easy identification
    # - draw_volume=False: Uses 2D representation instead of 3D for clarity
    # - scale_xy=1.5: Makes the visualization larger for better readability
    # - scale_z=1: Keeps depth scaling normal
    # - spacing=20: Adds space between layers for visual separation
    #
    # The generated image will show:
    # 1. Input layer (token sequences)
    # 2. Embedding layer (token to vector conversion)
    # 3. Transformer blocks (attention + feed-forward layers)
    # 4. Output layer (vocabulary predictions)
    arch_path = 'transformer_text_model_architecture.png'
    try:
        visualkeras.layered_view(
            model,
            to_file=arch_path,
            legend=True,          # Show layer names and types
            draw_volume=False,    # Use 2D representation for clarity
            scale_xy=1.5,         # Scale factor for x,y dimensions (larger = more readable)
            scale_z=1,            # Scale factor for z dimension (depth)
            spacing=20            # Spacing between layers in pixels
        )
        print(f"✅ Model architecture visualization saved to: {arch_path}")
        print("   This image shows the complete Transformer architecture with all layers")
    except Exception as e:
        print(f"❌ Could not generate VisualKeras visualization: {e}")
        print("   Note: VisualKeras requires additional dependencies (PIL, aggdraw)")

    # ========================================================================
    # STEP 5: SETUP TRAINING MONITORING AND LOGGING
    # ========================================================================
    print("\nSetting up training monitoring...")
    
    # CREATE LOGGING DIRECTORY:
    # ========================
    # Create a unique directory for this training run's logs and outputs
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    
    # TENSORBOARD CALLBACK:
    # ====================
    # TensorBoard provides real-time monitoring of training metrics
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,     # Log weight histograms every epoch
        write_graph=True,     # Log the computational graph
        write_images=True,    # Log model weights as images
        update_freq='epoch'   # Update logs every epoch
    )
    
    print(f"TensorBoard logs will be saved to: {os.path.abspath(logdir)}")
    print(f"To view training progress, run: tensorboard --logdir {logdir}")

    # VISUALKERAS INTEGRATION WITH TENSORBOARD:
    # =========================================
    # Log the VisualKeras architecture image to TensorBoard for easy viewing alongside training metrics.
    # This allows you to see the model architecture directly in the TensorBoard interface under the "Images" tab.
    #
    # PROCESS:
    # 1. Read the saved VisualKeras PNG file from disk
    # 2. Decode it as a 4-channel PNG image (RGBA)
    # 3. Add a batch dimension (TensorBoard expects batch format)
    # 4. Log it as a TensorFlow summary image with step=0 (before training)
    #
    # BENEFITS:
    # - Centralized viewing: Architecture and training metrics in one place
    # - Easy sharing: Send TensorBoard logs to colleagues with everything included
    # - Documentation: Permanent record of model architecture used for this run
    try:
        with tf.summary.create_file_writer(logdir).as_default():
            # Read the VisualKeras image file as raw bytes
            img = tf.io.read_file(arch_path)
            # Decode PNG with 4 channels (RGBA: Red, Green, Blue, Alpha)
            img = tf.image.decode_png(img, channels=4)
            # Add batch dimension and log to TensorBoard
            tf.summary.image("Model Architecture Visualization", tf.expand_dims(img, 0), step=0)
        print("✅ Model visualization logged to TensorBoard (Images tab)")
        print("   View it at: http://localhost:6006/#images after starting TensorBoard")
    except Exception as e:
        print(f"❌ Could not log VisualKeras image to TensorBoard: {e}")

    # ========================================================================
    # STEP 6: TRAIN THE MODEL
    # ========================================================================
    print("\nStarting model training...")
    
    # EARLY STOPPING CALLBACK:
    # ========================
    # Prevents overfitting by stopping training when loss stops improving
    early_stopping = EarlyStopping(
        monitor='loss',              # Metric to monitor
        patience=2,                  # Number of epochs to wait for improvement
        restore_best_weights=True    # Restore weights from best epoch
    )
    
    # MODEL TRAINING:
    # ==============
    # Train the model using the prepared sequences
    history = model.fit(
        X, Y,                                           # Training data
        epochs=20,                                      # Maximum number of training epochs
        batch_size=32,                                  # Number of samples per batch
        callbacks=[early_stopping, tensorboard_cb]     # Training callbacks
    )
    
    print("Training completed!")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")

    # ========================================================================
    # STEP 7: VISUALIZE TRAINING PROGRESS
    # ========================================================================
    print("\nCreating training visualizations...")
    
    # PLOT TRAINING LOSS:
    # ==================
    # Create a plot showing how the loss decreased during training
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], linewidth=2, color='blue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Save the plot to the logging directory
    plot_path = os.path.join(logdir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training loss plot saved to: {plot_path}")

    # ========================================================================
    # STEP 8: GENERATE TEXT WITH THE TRAINED MODEL
    # ========================================================================
    print("\n" + "="*60)
    print("GENERATING TEXT WITH TRAINED MODEL")
    print("="*60)
    
    # STANDARD TEXT GENERATION:
    # ========================
    print("\nGenerating text (100 tokens)...")
    start_string = "To be, or not to be"
    
    generated_text = generate_text(
        model=model,
        vectorizer=vectorizer,
        start_string=start_string,
        seq_length=seq_length,
        num_generate=100,
        temperature=0.7  # Slightly focused sampling for coherent text
    )
    
    print("\nGenerated text (100 tokens):")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

    # LONGER TEXT GENERATION:
    # ======================
    print("\nGenerating longer text sequence (200 tokens)...")
    
    longer_text = generate_text(
        model=model,
        vectorizer=vectorizer,
        start_string=start_string,
        seq_length=seq_length,
        num_generate=200,
        temperature=0.8  # Slightly more creative for longer text
    )
    
    print("\nGenerated text (200 tokens):")
    print("-" * 40)
    print(longer_text)
    print("-" * 40)

    # TEMPERATURE COMPARISON:
    # ======================
    print("\nGenerating text with different temperatures for comparison...")
    
    temperatures = [0.5, 1.0, 1.5]
    temp_results = {}
    
    for temp in temperatures:
        temp_text = generate_text(
            model=model,
            vectorizer=vectorizer,
            start_string=start_string,
            seq_length=seq_length,
            num_generate=50,
            temperature=temp
        )
        temp_results[temp] = temp_text
        print(f"\nTemperature {temp}:")
        print(temp_text[:200] + "...")  # Show first 200 characters

    # ========================================================================
    # STEP 9: SAVE RESULTS AND OUTPUTS
    # ========================================================================
    print("\nSaving results...")
    
    # SAVE GENERATED TEXT TO FILE:
    # ============================
    text_output_path = os.path.join(logdir, 'generated_text.txt')
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write("TRANSFORMER TEXT GENERATION RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Model Configuration:\n")
        f.write(f"- Vocabulary Size: {vocab_size}\n")
        f.write(f"- Sequence Length: {seq_length}\n")
        f.write(f"- Embedding Dimension: {embed_dim}\n")
        f.write(f"- Number of Heads: {num_heads}\n")
        f.write(f"- Feed-Forward Dimension: {ff_dim}\n")
        f.write(f"- Number of Layers: {num_layers}\n")
        f.write(f"- Total Parameters: {model.count_params():,}\n\n")
        
        f.write(f"Start String: '{start_string}'\n\n")
        
        f.write("Generated Text (100 tokens):\n")
        f.write("-" * 30 + "\n")
        f.write(generated_text + "\n\n")
        
        f.write("Generated Text (200 tokens):\n")
        f.write("-" * 30 + "\n")
        f.write(longer_text + "\n\n")
        
        f.write("Temperature Comparison:\n")
        f.write("-" * 30 + "\n")
        for temp, text in temp_results.items():
            f.write(f"Temperature {temp}:\n{text}\n\n")
    
    print(f"Generated text saved to: {text_output_path}")
    
    # SAVE MODEL WEIGHTS:
    # ==================
    model_path = os.path.join(logdir, 'transformer_model.h5')
    model.save_weights(model_path)
    print(f"Model weights saved to: {model_path}")
    
    # FINAL SUMMARY:
    # =============
    print("\n" + "="*60)
    print("TRAINING AND GENERATION COMPLETE!")
    print("="*60)
    print(f"📁 All outputs saved to: {os.path.abspath(logdir)}")
    print(f"📊 View training progress: tensorboard --logdir {logdir}")
    print(f"📝 Generated text available in: generated_text.txt")
    print(f"💾 Model weights saved for future use")
    print("="*60)


if __name__ == "__main__":
    # Entry point: run the main function when script is executed directly
    main() 