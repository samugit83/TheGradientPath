#!/usr/bin/env python3
"""
Complete script to generate synthetic stock price data, build a Transformer-based time series forecasting model,
train it with TensorBoard and Visualkeras integration, evaluate its performance, and visualize predictions.
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import visualkeras

# Configure TensorFlow to use GPU if available
# This section checks if a GPU is available and configures TensorFlow to use it.
# Using a GPU can significantly speed up model training.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once.
        # This is useful when sharing the GPU with other processes.
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


class MultiHeadSelfAttention(Layer):
    """
    Implements the Multi-Head Self-Attention mechanism.
    Self-attention allows the model to weigh the importance of different parts of the input sequence
    when processing a particular part of the sequence. Think of it like the model paying "attention"
    to relevant words in a sentence to understand the context of a specific word.
    Multi-Head means we do this attention process multiple times in parallel (with different "heads")
    and then combine the results. This allows the model to capture different types of relationships.
    """
    def __init__(self, embed_dim, num_heads=8):
        """
        Initializes the MultiHeadSelfAttention layer.
        Args:
            embed_dim (int): The dimensionality of the input and output embeddings.
                             This is the size of the vector representing each element in the sequence.
            num_heads (int): The number of attention heads. More heads can capture more complex patterns
                             but also increase computation.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # The projection dimension for each head is calculated by dividing the embedding dimension
        # by the number of heads. This ensures that the total computation remains manageable.
        self.projection_dim = embed_dim // num_heads
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Dense layers to project the input into Query (Q), Key (K), and Value (V) spaces.
        # - Query: Represents the current element we are focusing on.
        # - Key: Represents all elements in the sequence that we compare against the Query.
        # - Value: Represents the information carried by each element in the sequence.
        # The model learns the weights of these Dense layers during training.

        
        self.query_dense = Dense(embed_dim)  # Projects input to Query
        self.key_dense = Dense(embed_dim)    # Projects input to Key
        self.value_dense = Dense(embed_dim)  # Projects input to Value
        # The Dense layers for Query, Key, and Value projections operate on each time step independently.
        # For each position in the sequence, they transform the embedding vector (of size embed_dim)
        # into new vectors of the same size.
        #
        # For example, if your input has shape (batch_size, seq_len, embed_dim):
        # - Each Dense layer processes each of the seq_len vectors (of size embed_dim) independently
        # - The same weights are used for all positions in the sequence
        # - The output shape remains (batch_size, seq_len, embed_dim)
        #
        # This is different from flattening the sequence and embedding dimensions together.
        # Instead, it's like having a separate transformation for each position in the sequence,
        # but with shared weights across all positions.


        # Dense layer to combine the outputs of all attention heads.
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        """
        Calculates the attention scores and output.
        This is the core of the self-attention mechanism.
        Args:
            query: Projected query tensor.
            key: Projected key tensor.
            value: Projected value tensor.
        Returns:
            output: The context vector, which is a weighted sum of Values.
            weights: The attention weights, indicating the importance of each Value.
        """
        # 1. Calculate similarity scores: MatMul(Query, Key_transposed)
        # This measures how much each Query matches each Key.
        # The dot product is implemented here using matrix multiplication (tf.matmul)
        # Each element in the resulting score matrix is the dot product between a query vector
        # and a key vector, representing their similarity

        # Let's understand the shapes and the dot product operation:
        # 
        # At this point, query and key have been processed through split_heads(), so they have shape:
        # query: (batch_size, num_heads, seq_len, projection_dim)
        # key:   (batch_size, num_heads, seq_len, projection_dim)
        # 
        # where projection_dim = embed_dim // num_heads
        #
        # The dot product operation tf.matmul(query, key, transpose_b=True) works as follows:
        # 
        # 1. transpose_b=True means we transpose the last two dimensions of the 'key' tensor
        #    So key becomes: (batch_size, num_heads, projection_dim, seq_len)
        #    
        # 2. The matrix multiplication is performed on the last two dimensions:
        #    query: (..., seq_len, projection_dim) × key_transposed: (..., projection_dim, seq_len)
        #    Result: (..., seq_len, seq_len)
        #
        # 3. For each batch and each head, we're computing:
        #    - query[i] (a vector of size projection_dim at position i) 
        #    - dot product with key[j] (a vector of size projection_dim at position j)
        #    - This gives us a similarity score between position i and position j
        #
        # 4. The resulting score matrix has shape (batch_size, num_heads, seq_len, seq_len)
        #    where score[b, h, i, j] = similarity between query at position i and key at position j
        #    for batch b and attention head h
        #
        # Example with concrete numbers:
        # If seq_len=100, embed_dim=128, num_heads=8, then projection_dim=16
        # query: (batch_size, 8, 100, 16)
        # key:   (batch_size, 8, 100, 16) -> after transpose_b: (batch_size, 8, 16, 100)
        # score: (batch_size, 8, 100, 100) - a 100x100 attention matrix for each head
        score = tf.matmul(query, key, transpose_b=True)  # This is the dot product operation
        # Shape of score: (batch_size, num_heads, seq_len, seq_len)
        # Each element score[b, h, i, j] represents the similarity between query at position i
        # and key at position j for batch b and head h

        # 2. Scale the scores: score / sqrt(dimension_of_key)
        # Scaling prevents the dot products from becoming too large, which can lead to
        # vanishing gradients in the softmax function.
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # 3. Apply softmax to get attention weights: Softmax(scaled_score)
        # Softmax converts the scores into probabilities (summing to 1), representing
        # the "attention" or importance of each Value for a given Query.
        weights = tf.nn.softmax(scaled_score, axis=-1)
        # Shape of weights: (batch_size, num_heads, seq_len, seq_len)
        # Each element weights[b, h, i, j] represents the attention weight that position i
        # gives to position j for batch b and head h. The weights along the last dimension
        # (axis=-1) sum to 1.0, forming a probability distribution over all positions.

        # 4. Calculate the output: MatMul(weights, Value)
        # The output is a weighted sum of the Values, where the weights are determined by
        # the attention mechanism. This gives a context vector that emphasizes important parts.
        # output type: Tensor with shape (batch_size, num_heads, seq_len, projection_dim)
        # This tensor contains the contextualized representations for each position in the sequence
        # This is a weighted sum of values, where weights come from attention scores
        output = tf.matmul(weights, value) # Shape: (batch_size, num_heads, seq_len, projection_dim)
        return output, weights

    def split_heads(self, x, batch_size):
        """
        Splits the input tensor into multiple heads.
        This allows each head to attend to different parts of the input independently.
        Args:
            x: Input tensor (batch_size, sequence_length, embed_dim).
            batch_size: The size of the input batch.
        Returns:
            Tensor reshaped for multi-head attention: (batch_size, num_heads, sequence_length, projection_dim).
        """
        # Reshape the input to (batch_size, seq_len, num_heads, projection_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        # Transpose to (batch_size, num_heads, seq_len, projection_dim)
        # This groups all data for a single head together, which is needed for parallel computation.
        # perm is the permutation parameter that specifies the new order of dimensions
        # [0, 2, 1, 3] means:
        # - 0th dimension (batch_size) stays in position 0
        # - 2nd dimension (num_heads) moves to position 1
        # - 1st dimension (seq_len) moves to position 2
        # - 3rd dimension (projection_dim) stays in position 3
        # This rearranges from (batch_size, seq_len, num_heads, projection_dim)
        # to (batch_size, num_heads, seq_len, projection_dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """
        Forward pass of the MultiHeadSelfAttention layer.
        Args:
            inputs: The input sequence tensor (batch_size, sequence_length, embed_dim).
        Returns:
            The output tensor after applying multi-head self-attention
            (batch_size, sequence_length, embed_dim).
        """
        batch_size = tf.shape(inputs)[0]

        # 1. Project inputs into Q, K, V
        # BEFORE: Each input element is projected to three different representations
        #
        # HOW PERCEPTRONS ARE CONNECTED FROM PREVIOUS DENSE TO Q, K, V:
        # Input from previous Dense layer: (batch_size, seq_len, embed_dim)
        # - Each time step has embed_dim features (e.g., 128 features per time step)
        # - These come from the initial Dense(embed_dim) projection of the time series data
        #
        # For EACH of the Q, K, V projections:
        # Each output neuron is FULLY CONNECTED to ALL embed_dim input features
        #
        # Connection pattern for each time step t:
        # Previous Dense output: [emb_0, emb_1, emb_2, ..., emb_127] (embed_dim=128)
        #                            ↓     ↓     ↓         ↓
        # Query Dense (embed_dim neurons):
        #   query_neuron_0 ←─────┬─────┬─────┬─────...─────┬ (connected to ALL embed_dim inputs)
        #   query_neuron_1 ←─────┼─────┼─────┼─────...─────┼ (connected to ALL embed_dim inputs)
        #   query_neuron_2 ←─────┼─────┼─────┼─────...─────┼ (connected to ALL embed_dim inputs)
        #   ...                  │     │     │             │
        #   query_neuron_127 ←───┴─────┴─────┴─────...─────┴ (connected to ALL embed_dim inputs)
        #
        # Key Dense (embed_dim neurons):
        #   key_neuron_0 ←───────┬─────┬─────┬─────...─────┬ (connected to ALL embed_dim inputs)
        #   key_neuron_1 ←───────┼─────┼─────┼─────...─────┼ (connected to ALL embed_dim inputs)
        #   ...                  │     │     │             │
        #   key_neuron_127 ←─────┴─────┴─────┴─────...─────┴ (connected to ALL embed_dim inputs)
        #
        # Value Dense (embed_dim neurons):
        #   value_neuron_0 ←─────┬─────┬─────┬─────...─────┬ (connected to ALL embed_dim inputs)
        #   value_neuron_1 ←─────┼─────┼─────┼─────...─────┼ (connected to ALL embed_dim inputs)
        #   ...                  │     │     │             │
        #   value_neuron_127 ←───┴─────┴─────┴─────...─────┴ (connected to ALL embed_dim inputs)
        #
        # Each Dense layer has: embed_dim × embed_dim weights + embed_dim biases
        # Total parameters per Dense layer: embed_dim × (embed_dim + 1)
        # For embed_dim=128: 128 × 129 = 16,512 parameters per Q/K/V layer
        #
        # Mathematical operation for each neuron i in each layer:
        # query[t,i] = Σ(j=0 to embed_dim-1) W_query[i,j] × input[t,j] + bias_query[i]
        # key[t,i]   = Σ(j=0 to embed_dim-1) W_key[i,j] × input[t,j] + bias_key[i]
        # value[t,i] = Σ(j=0 to embed_dim-1) W_value[i,j] × input[t,j] + bias_value[i]
        #
        # The SAME weights are shared across ALL time steps (just like before)
        # But each Q/K/V projection learns DIFFERENT transformations of the same input
        #
        # HOW SEQUENCE LENGTH (seq_len) IS MANAGED:
        # The Dense layers do NOT flatten the sequence dimension!
        # Instead, they apply the SAME transformation to EACH time step INDEPENDENTLY
        #
        # Think of it as seq_len PARALLEL applications of the same Dense layer:
        #
        # Time Step 0: [emb_0_0, emb_0_1, ..., emb_0_127] ──> Dense ──> [q_0_0, q_0_1, ..., q_0_127]
        # Time Step 1: [emb_1_0, emb_1_1, ..., emb_1_127] ──> Dense ──> [q_1_0, q_1_1, ..., q_1_127]
        # Time Step 2: [emb_2_0, emb_2_1, ..., emb_2_127] ──> Dense ──> [q_2_0, q_2_1, ..., q_2_127]
        # ...          ...                                      ↑        ...
        # Time Step 99:[emb_99_0,emb_99_1,...,emb_99_127] ──> Same ──> [q_99_0,q_99_1,...,q_99_127]
        #                                                     Weights
        #
        # Key points about seq_len management:
        # 1. NO connections between different time steps at this stage
        # 2. Each time step processed with IDENTICAL weights and biases
        # 3. The Dense layer sees each time step as a separate "sample"
        # 4. seq_len = 100 means we apply the same transformation 100 times in parallel
        #
        # Shape transformations:
        # Input:  (batch_size, seq_len, embed_dim) = (batch_size, 100, 128)
        # Dense layer treats this as: (batch_size × seq_len, embed_dim) = (batch_size × 100, 128)
        # Applies transformation to each of the (batch_size × 100) vectors of size 128
        # Output: (batch_size × seq_len, embed_dim) = (batch_size × 100, 128)
        # Reshaped back to: (batch_size, seq_len, embed_dim) = (batch_size, 100, 128)
        #
        # This is why we can have ANY sequence length - the Dense layer doesn't care!
        # It just applies the same transformation to however many time steps you give it.
        query = self.query_dense(inputs)  # Shape: (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)      # Shape: (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # Shape: (batch_size, seq_len, embed_dim)
        # At this stage, we have single representations of Q, K, V with full embedding dimension

        # 2. Split Q, K, V into multiple heads
        # AFTER: We divide the embedding dimension into smaller chunks for parallel processing
        # The embed_dim (e.g., 128) is split into num_heads (e.g., 8) pieces, each with projection_dim (e.g., 16)
        query = self.split_heads(query, batch_size)  # Shape: (batch_size, num_heads, seq_len, projection_dim)
        key = self.split_heads(key, batch_size)      # Shape: (batch_size, num_heads, seq_len, projection_dim)
        value = self.split_heads(value, batch_size)  # Shape: (batch_size, num_heads, seq_len, projection_dim)
        # Now each head can focus on different aspects of the data in parallel
        # Think of it like having multiple experts analyzing the same sequence from different perspectives

        # 3. Calculate attention for each head
        # The attention function is applied in parallel to all heads.
        attention_output, _ = self.attention(query, key, value)  # (batch_size, num_heads, seq_len, projection_dim)

        # 4. Concatenate attention heads
        # First, transpose back to (batch_size, seq_len, num_heads, projection_dim)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        # Then, reshape to (batch_size, seq_len, embed_dim) to combine the information from all heads.
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))

        # 5. Apply the final linear projection
        # This allows the model to learn how to best combine the outputs of the different heads.
        # This combines all the attention head outputs into a single representation
        return self.combine_heads(concat_attention) # Final output shape: (batch_size, seq_len, embed_dim)


class TransformerBlock(Layer):
    """
    Implements a single Transformer Block.
    A Transformer Block is a fundamental building unit of the Transformer model.
    It consists of:
    1. Multi-Head Self-Attention: To understand relationships within the input sequence.
    2. Feed-Forward Network (FFN): A simple position-wise neural network to process the output of attention.
    3. Layer Normalization: Applied before and after each sub-layer to stabilize training.
    4. Dropout: Applied to prevent overfitting.
    5. Residual Connections: Adds the input of a sub-layer to its output (input + SubLayer(input)).
       This helps with training deep networks by allowing gradients to flow more easily.
    """ 
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initializes the TransformerBlock.
        Args:
            embed_dim (int): Dimensionality of the input and output embeddings.
            num_heads (int): Number of attention heads for the MultiHeadSelfAttention layer.
            ff_dim (int): Dimensionality of the inner layer of the Feed-Forward Network.
                          Typically, this is larger than embed_dim (e.g., 4 * embed_dim).
            rate (float): Dropout rate.
        """
        super(TransformerBlock, self).__init__()
        # Multi-Head Self-Attention layer
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        # Position-wise Feed-Forward Network (FFN)
        # It consists of two Dense layers with a ReLU activation in between.
        # This FFN is applied independently to each position in the sequence.
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),  # Expands dimensionality
            Dense(embed_dim),                 # Projects back to original dimensionality
        ])
        # Layer Normalization: Normalizes the activations within each layer.
        # This helps stabilize training and often leads to faster convergence.
        self.layernorm1 = LayerNormalization(epsilon=1e-6) # Epsilon for numerical stability
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        # Dropout: Randomly sets a fraction of input units to 0 during training.
        # This is a regularization technique to prevent overfitting.
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        """
        Forward pass of the TransformerBlock.
        Args:
            inputs: Input tensor (batch_size, sequence_length, embed_dim).
            training (bool): Whether the model is in training mode (True) or inference mode (False).
                             Dropout is only applied during training.
        Returns:
            Output tensor of the Transformer Block (batch_size, sequence_length, embed_dim).
        """
        # 1. MULTI-HEAD SELF-ATTENTION SUB-LAYER
        # =======================================
        # CONCEPTUAL PURPOSE: The attention mechanism allows each position in the sequence
        # to "look at" and gather information from all other positions. This creates
        # context-aware representations where each element understands its relationship
        # to the entire sequence.
        #
        # WHY THIS MATTERS: In time series, this means each time step can consider
        # patterns and dependencies from all other time steps, not just adjacent ones.
        # For example, a spike at time t=10 might be related to a pattern at t=3,
        # and attention can capture this long-range dependency.
        #
        # DETAILED MECHANICS:
        # - self.att(inputs) calls the multi-head self-attention layer on inputs
        # - Inside that, each token in the sequence looks at every other token (including itself)
        # - This produces a new representation for each position by mixing all tokens' "value" 
        #   vectors according to their attention weights
        # - Each slice attn_output[b, i, :] is a "contextualized" embedding for token i
        #
        # inputs shape: (batch_size, sequence_length, embed_dim)
        attn_output = self.att(inputs)  # attn_output shape: (batch_size, sequence_length, embed_dim)
        
        # DROPOUT ON ATTENTION OUTPUT:
        # - self.dropout1() randomly zeroes out a fraction (dropout_rate) of elements in attn_output
        # - Only applied during training (when training=True)
        # - If training=False, dropout does nothing and simply returns the input unchanged
        # - PURPOSE: Prevents overfitting by ensuring the model can't rely too heavily on 
        #   any single dimension in the attention output
        # - Forces the model to learn more robust representations that don't depend on 
        #   specific attention patterns
        attn_output = self.dropout1(attn_output, training=training)
        
        # FIRST RESIDUAL CONNECTION + LAYER NORMALIZATION:
        # ================================================
        # RESIDUAL CONNECTION (inputs + attn_output):
        # - Element-wise sum of original inputs and attention output
        # - Both have shape: (batch_size, sequence_length, embed_dim)
        # - Each position's new representation = original embedding + attention-modified embedding
        # - BENEFITS:
        #   * Gradients can flow directly through the network during backpropagation
        #   * Prevents vanishing gradients in deep networks
        #   * Provides "highway" for information that doesn't need attention modification
        #   * Allows the block to learn identity mapping if attention isn't helpful
        #
        # LAYER NORMALIZATION:
        # - Normalizes across the embed_dim dimension for each position (b, i)
        # - For each token position i in each batch example b:
        #   1. Computes mean and variance of vector (inputs + attn_output)[b, i, :] 
        #      over its embed_dim components
        #   2. Subtracts mean and divides by standard deviation (+ epsilon for stability)
        #   3. Applies learned scale (γ) and shift (β) parameters of size embed_dim
        # - RESULT: out1 has zero mean, unit variance per position → stable training
        # - WHY: Prevents activations from becoming too large/small, speeds convergence
        out1 = self.layernorm1(inputs + attn_output)

        # 2. FEED-FORWARD NETWORK (FFN) SUB-LAYER
        # ========================================
        # CONCEPTUAL PURPOSE: While attention captures relationships between positions,
        # the FFN processes each position independently with non-linear transformations.
        # This adds computational capacity and allows learning complex feature 
        # transformations at each position.
        #
        # WHY THIS MATTERS: The FFN acts like a position-wise "feature processor"
        # that can enhance, suppress, or transform the attended features. It's like
        # having a small neural network at each time step that recognizes and 
        # transforms patterns in the context-aware representations from attention.
        #
        # DETAILED MECHANICS:
        # - self.ffn is a two-layer, position-wise feed-forward network:
        #   * Dense(ff_dim, activation="relu")  # expands to ff_dim (e.g., 4x embed_dim)
        #   * Dense(embed_dim)                  # projects back to embed_dim
        # - Applied independently at each token position:
        #   1. Take vector out1[b, i, :] of length embed_dim
        #   2. Run through dense layer → ff_dim size, apply ReLU
        #   3. Run through second dense layer → back to embed_dim size
        # - Because applied independently per position, output shape unchanged
        # - ARCHITECTURE RATIONALE:
        #   * First layer expands dimensionality for more representational capacity
        #   * ReLU adds non-linearity for complex transformations
        #   * Second layer compresses back to match residual connection requirements
        ffn_output = self.ffn(out1)  # shape: (batch_size, sequence_length, embed_dim)
        
        # DROPOUT ON FFN OUTPUT:
        # - Again applies dropout to FFN's output during training
        # - Zeroes out dropout_rate fraction of elements
        # - PURPOSE: Further regularizes the model, prevents overfitting to specific
        #   FFN transformations
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # SECOND RESIDUAL CONNECTION + LAYER NORMALIZATION:
        # =================================================
        # RESIDUAL CONNECTION (out1 + ffn_output):
        # - Element-wise sum of FFN input (out1) and FFN output (ffn_output)
        # - Both have shape: (batch_size, sequence_length, embed_dim)
        # - BENEFITS: Same as first residual connection
        #   * Gradient flow for deep network training
        #   * Allows network to "choose" whether to modify representation via FFN
        #   * Can preserve information if FFN transformation isn't beneficial
        #
        # LAYER NORMALIZATION:
        # - Same process as before but with separate learned scale/shift parameters
        # - Normalizes across embedding dimension for each position
        # - Ensures stable distribution for next layer or final output
        #
        # FINAL RESULT: Each Transformer block achieves two key capabilities:
        # (a) Mix information across all positions via attention
        # (b) Transform each position's representation via position-wise MLP
        # While residual connections and layer norms ensure stable gradients and training
        return self.layernorm2(out1 + ffn_output)


class TransformerEncoder(Layer):
    """
    Implements the Transformer Encoder.
    The Encoder is a stack of multiple Transformer Blocks.
    It takes a sequence of embeddings as input and produces a sequence of context-aware embeddings.
    Each block processes the output of the previous block, allowing the model to build
    increasingly complex representations of the input sequence.
    """
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initializes the TransformerEncoder.
        Args:
            num_layers (int): The number of Transformer Blocks to stack.
            embed_dim (int): Dimensionality of embeddings.
            num_heads (int): Number of attention heads in each Transformer Block.
            ff_dim (int): Dimensionality of the FFN inner layer in each Transformer Block.
            rate (float): Dropout rate.
        """
        super(TransformerEncoder, self).__init__()
        # Create a list of 'num_layers' TransformerBlocks.
        self.enc_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate)
                           for _ in range(num_layers)]
        # Dropout layer to be applied to the input of the encoder (optional, sometimes applied to output)
        self.dropout = Dropout(rate) # Note: This dropout is on the input to the first block, not after all blocks.
                                     # Positional encoding is typically added before this if used.

    def call(self, inputs, training=False):
        """
        Forward pass of the TransformerEncoder.
        Args:
            inputs: Input sequence tensor (batch_size, sequence_length, embed_dim).
                    In this model, 'inputs' are the time series data after an initial Dense projection.
            training (bool): Whether the model is in training mode.
        Returns:
            Output tensor of the Transformer Encoder (batch_size, sequence_length, embed_dim).
        """
        x = inputs # Start with the input
        # Note: In a standard Transformer, positional encodings would be added to 'x' here
        # to give the model information about the order of elements in the sequence.
        # This example omits explicit positional encoding, relying on the Dense layer
        # and the sequence structure itself for some positional awareness.

        # DROPOUT ON INPUT:
        # - self.dropout() randomly zeroes out a fraction (dropout_rate) of elements in 'x'
        # - Only applied during training (when training=True)
        # - If training=False, dropout does nothing and simply returns the input unchanged
        # - PURPOSE: Prevents overfitting by ensuring the model can't rely too heavily on 
        #   any single dimension in the input
        x = self.dropout(x, training=training)

        

        # Pass the input through each Transformer Block in the stack.
        for layer in self.enc_layers:
            x = layer(x, training=training)
        return x  # shape: (batch_size, sequence_length, embed_dim)


def create_dataset(data, time_step=1):
    """
    Prepares the dataset for time series forecasting.
    Converts a time series into sequences of a fixed 'time_step' length (features)
    and the next value in the series (target).
    Example:
        data = [10, 20, 30, 40, 50, 60], time_step = 3
        X (features) will be:
            [[10, 20, 30],
             [20, 30, 40],
             [30, 40, 50]]
        Y (target) will be:
            [40,
             50,
             60]
    Args:
        data (np.array): The input time series data (a single column of values).
        time_step (int): The number of past time steps to use as input features.
    Returns:
        X (np.array): Array of input sequences.
        Y (np.array): Array of target values.
    """
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        # Take 'time_step' elements as features
        X.append(data[i:(i + time_step), 0])
        # Take the element immediately following the feature sequence as the target
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


def build_model(time_step,
                embed_dim=128,
                num_heads=8,
                ff_dim=512,
                num_layers=4,
                dropout_rate=0.1):
    """
    Build and return a Transformer-based time series forecasting model.
    The model architecture:
    1. Input Layer: Takes sequences of 'time_step' length.
    2. Dense Layer: Projects the input features (each point in the time_step) into a higher-dimensional
                   embedding space ('embed_dim'). This is where the model starts to learn representations.
                   This also serves as a form of "embedding" for the time series values.
    3. Transformer Encoder: Processes the sequence of embeddings to capture temporal dependencies
                           and relationships using self-attention.
    4. Flatten Layer: Flattens the output of the Transformer Encoder to prepare it for the final Dense layer.
                     The encoder outputs a sequence of embeddings (one for each time step in the input sequence),
                     so we flatten it to a single vector.
    5. Dropout Layer: Regularization to prevent overfitting before the final prediction.
    6. Dense Output Layer: A single neuron to predict the next value in the time series.
    Args:
        time_step (int): Number of past time steps to use as input.
        embed_dim (int): Dimensionality of the embeddings used within the Transformer.
        num_heads (int): Number of attention heads in the Transformer blocks.
        ff_dim (int): Dimensionality of the feed-forward network's inner layer in Transformer blocks.
        num_layers (int): Number of Transformer blocks in the encoder.
        dropout_rate (float): Dropout rate for regularization.
    Returns:
        A compiled Keras Model.
    """
    # Input layer: expects sequences of shape (time_step, 1)
    # Each sample is a sequence of 'time_step' observations, and each observation has 1 feature (the stock price).
    # Input layer: shape=(time_step, 1) defines the shape of a single sample
    # The batch_size is not specified here and is handled automatically by Keras
    # during model training (typically as the first dimension of the input tensor)
    inputs = Input(shape=(time_step, 1))  # Actual shape during training: (batch_size, time_step, 1)

    # Initial Dense layer to project input features into the embedding dimension.
    # This can be seen as creating an "embedding" for each time step in the input sequence.
    # The Transformer works with these embeddings.
    #
    # HOW THE PERCEPTRONS ARE CONNECTED:
    # Input shape: (batch_size, time_step, 1) - each time step has 1 feature
    # Dense(embed_dim) creates embed_dim neurons/perceptrons
    # 
    # Connection pattern for each time step:
    # - Each of the embed_dim output neurons is connected to the single input feature
    # - Each neuron has: 1 weight (for the input feature) + 1 bias = 2 parameters per neuron
    # - Total parameters: embed_dim * (1 + 1) = embed_dim * 2
    #
    # Mathematical operation for each time step t:
    # input_t = inputs[:, t, 0]  # shape: (batch_size,) - single feature value
    # For each output neuron i (i = 0 to embed_dim-1):
    #   output_t[i] = weight_i * input_t + bias_i
    #
    # The SAME weights and biases are used for ALL time steps (weight sharing)
    # This is equivalent to applying the same linear transformation at each time step
    #
    # Visual representation for embed_dim=3:
    # Time step t:
    #   input_feature ──┬──> [neuron_0] ──> output_0
    #                   ├──> [neuron_1] ──> output_1  
    #                   └──> [neuron_2] ──> output_2
    #
    # This pattern repeats for every time step in the sequence
    x = Dense(embed_dim)(inputs)  # Output shape: (batch_size, time_step, embed_dim)


    # Example: Time step 50 might learn that time steps 45-49 and 51-55 are very relevant,
    # while time steps 0-10 and 90-99 are less relevant for its processing.
    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, rate=dropout_rate)
    x = encoder(x)  # Output shape: (batch_size, time_step, embed_dim)

    # Flatten the output of the encoder to prepare for final prediction
    # ================================================================
    # WHAT THE ENCODER OUTPUTS:
    # The TransformerEncoder produces a 3D tensor with shape (batch_size, time_step, embed_dim)
    # This means for each sample in the batch, we have 'time_step' number of vectors,
    # where each vector has 'embed_dim' dimensions (e.g., 100 time steps × 128 dimensions each)
    #
    # WHY WE NEED TO FLATTEN:
    # Our final goal is to predict a single value (the next stock price), but we currently
    # have a sequence of vectors. We need to convert this 2D sequence of embeddings into
    # a single 1D vector that captures all the temporal information.
    #
    # WHAT FLATTEN DOES:
    # - Takes input shape: (batch_size, time_step, embed_dim)
    # - Reshapes to: (batch_size, time_step * embed_dim)
    # - Example: (32, 100, 128) becomes (32, 12800)
    # - This concatenates all time step embeddings into one long vector per sample
    #
    # CONCEPTUAL MEANING:
    # Think of it like taking a book (sequence of pages/time steps) where each page
    # contains a summary (embedding) of that time period, and then laying out all
    # the page summaries in a single long line. This gives us one comprehensive
    # representation that contains information from all time steps.
    #
    # ALTERNATIVE APPROACHES:
    # Instead of Flatten(), we could use:
    # - GlobalAveragePooling1D(): Average all time step embeddings
    # - Take only the last time step: x[:, -1, :]
    # - Use attention pooling to weight different time steps
    # But Flatten() preserves all positional information, letting the final Dense
    # layer learn which parts of the sequence are most important for prediction.
    x = tf.keras.layers.Flatten()(x) # Output shape: (batch_size, time_step * embed_dim)

    # Dropout layer for regularization before the final output layer.
    x = Dropout(dropout_rate)(x)

    # Output Dense layer with a single neuron to predict the next single value in the time series.
    # No activation function is used here (linear activation), which is common for regression tasks.
    outputs = Dense(1)(x) # Output shape: (batch_size, 1)

    # Create and return the Keras Model
    return Model(inputs, outputs)


def main():
    # ========================================================================
    # MAIN FUNCTION: COMPLETE TRANSFORMER TIME SERIES FORECASTING PIPELINE
    # ========================================================================
    # This function demonstrates a complete end-to-end workflow for using
    # a Transformer model to predict stock prices. It includes data generation,
    # preprocessing, model building, training, evaluation, and visualization.
    
    # 1. Generate synthetic stock price data
    # =====================================
    # We create artificial stock price data for demonstration purposes.
    # In a real scenario, you would load actual historical stock data.
    np.random.seed(42)  # Set seed for reproducible results
    data_length = 2000  # Number of data points to generate
    
    # Create a synthetic stock price trend that goes from $100 to $200
    trend = np.linspace(100, 200, data_length)
    
    # Add random noise to make the data more realistic (stock prices fluctuate)
    noise = np.random.normal(0, 2, data_length)  # Mean=0, std=2
    
    # Combine trend and noise to create synthetic stock prices
    synthetic_data = trend + noise
    
    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(synthetic_data, columns=['Close'])

    # 2. Normalize data
    # =================
    # Neural networks work better with normalized data (values between 0 and 1).
    # This prevents features with larger scales from dominating the learning process.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)

    # 3. Prepare dataset for training
    # ===============================
    # Convert the time series into supervised learning format:
    # - X: sequences of past 'time_step' values
    # - Y: the next value to predict
    time_step = 100  # Use 100 previous time steps to predict the next one
    X, Y = create_dataset(scaled_data, time_step)
    
    # Reshape X to add a feature dimension (required for Transformer input)
    # Shape changes from (samples, time_steps) to (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 4. Build and compile model
    # ==========================
    # Create the Transformer model with specified hyperparameters
    model = build_model(time_step,
                        embed_dim=128,      # Dimension of embeddings
                        num_heads=8,        # Number of attention heads
                        ff_dim=512,         # Feed-forward network dimension
                        num_layers=4,       # Number of Transformer layers
                        dropout_rate=0.1)   # Dropout rate for regularization
    
    # Compile the model with optimizer, loss function, and metrics
    model.compile(
        optimizer='adam',  # Adaptive learning rate optimizer
        loss='mse',        # Mean Squared Error for regression
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    # Print model architecture summary
    model.summary()

    # 5. Setup TensorBoard and Visualkeras logging
    # =============================================
    # Create a unique directory for this training run's logs
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    
    # Generate and save model architecture visualization using Visualkeras
    arch_path = os.path.join(logdir, 'model_visualkeras.png')
    visualkeras.layered_view(
        model,
        to_file=arch_path,
        legend=True,          # Show layer names
        draw_volume=False,    # Don't show 3D volume representation
        scale_xy=1.5,         # Scale factor for x,y dimensions
        scale_z=1,            # Scale factor for z dimension
        spacing=20            # Spacing between layers
    )
    
    # Log the Visualkeras image to TensorBoard for easy viewing
    with tf.summary.create_file_writer(logdir).as_default():
        img = tf.io.read_file(arch_path)
        img = tf.image.decode_png(img, channels=4)
        tf.summary.image("Model Visualization", tf.expand_dims(img, 0), step=0)

    # Setup TensorBoard callback for monitoring training
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,     # Log weight histograms every epoch
        write_graph=True,     # Log the model graph
        write_images=True,    # Log model weights as images
        update_freq='epoch',  # Update logs every epoch
        profile_batch=1       # Profile the first batch for performance analysis
    )
    
    # Print instructions for viewing TensorBoard
    print(f"TensorBoard logs in: {os.path.abspath(logdir)}")
    print(f"Run: tensorboard --logdir {logdir}")

    # 6. Train model with TensorBoard callback
    # ========================================
    # Train the model and store training history
    history = model.fit(
        X,                      # Input sequences
        Y,                      # Target values
        epochs=20,              # Number of training epochs
        batch_size=32,          # Number of samples per batch
        validation_split=0.1,   # Use 10% of data for validation
        callbacks=[tensorboard_cb]  # Include TensorBoard logging
    )

    # 7. Evaluate model
    # =================
    # Evaluate the trained model on the same data (in practice, use separate test data)
    loss = model.evaluate(X, Y)
    print(f"Test loss (MSE): {loss:.6f}")

    # 8. Make predictions
    # ===================
    # Generate predictions using the trained model
    predictions = model.predict(X)
    
    # Convert predictions back to original scale (inverse of normalization)
    predictions = scaler.inverse_transform(predictions)

    # 9. Plot results
    # ===============
    # Create a visualization comparing true data with model predictions
    plt.figure(figsize=(10, 6))
    
    # Plot the original stock price data
    plt.plot(df['Close'], label='True Data')
    
    # Plot predictions (offset by time_step since we can't predict the first time_step values)
    plt.plot(np.arange(time_step, time_step + len(predictions)), predictions, label='Predictions')
    
    # Add labels and formatting
    plt.title('Transformer Time Series Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Save the plot to the same directory as TensorBoard logs
    plot_path = os.path.join(logdir, 'predictions_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    # Entry point: run the main function when script is executed directly
    main()

