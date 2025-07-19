#!/usr/bin/env python3
"""
Complete script to implement Transformers for text generation with KV Cache support.
Build, train, and evaluate an LLM using a Transformer model for text generation using TensorFlow and Keras.
"""
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, LayerNormalization, Dropout, Embedding, 
    TextVectorization
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import visualkeras
from helpers import create_sequences, inference, load_corpus

# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured.")
    except RuntimeError as e:
        print(f"Error setting up GPU memory growth: {e}")
else:
    print("No GPU detected, using CPU.")


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        # Apply causal mask for autoregressive generation
        if mask is not None:
            scaled_score += (mask * -1e9)
        
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, kv_cache=None, use_cache=False, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Compute Q, K, V for current inputs
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Split into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Handle KV cache
        if use_cache and kv_cache is not None:
            # Concatenate cached keys and values with new ones
            cached_key = kv_cache.get('key')
            cached_value = kv_cache.get('value')
            
            if cached_key is not None and cached_value is not None:
                key = tf.concat([cached_key, key], axis=2)
                value = tf.concat([cached_value, value], axis=2)
        
        # Create causal mask
        total_seq_len = tf.shape(key)[2]
        mask = tf.linalg.band_part(tf.ones((seq_len, total_seq_len)), -1, 0)
        mask = tf.where(mask == 0, 1.0, 0.0)
        
        attention_output, attention_weights = self.attention(query, key, value, mask)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        
        # Update cache
        new_cache = {'key': key, 'value': value} if use_cache else None
        
        return output, new_cache


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, kv_cache=None, use_cache=False, training=False):
        attn_output, new_cache = self.att(inputs, kv_cache=kv_cache, use_cache=use_cache, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)
        return output, new_cache


class TransformerModel(Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dense = Dense(vocab_size)

    def positional_encoding(self, seq_length, embed_dim):
        angle_rads = self.get_angles(np.arange(seq_length)[:, np.newaxis], 
                                   np.arange(embed_dim)[np.newaxis, :], embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def call(self, inputs, kv_cache=None, use_cache=False, start_pos=0, training=False):
        seq_len = tf.shape(inputs)[1]
        
        # Handle positional encoding based on start_pos
        if start_pos > 0 and start_pos < self.seq_length:
            # For decode phase, only get positional encoding for current positions
            pos_encoding = self.pos_encoding[:, start_pos:start_pos + seq_len, :]
        else:
            # For prefill phase, training, or when start_pos is out of bounds
            # Use the last positions of the positional encoding
            if start_pos >= self.seq_length:
                # Use the last position for tokens beyond the training sequence length
                pos_encoding = self.pos_encoding[:, -1:, :]
                pos_encoding = tf.tile(pos_encoding, [1, seq_len, 1])
            else:
                pos_encoding = self.pos_encoding[:, :seq_len, :]
        
        x = self.embedding(inputs)
        x += pos_encoding
        
        # Initialize cache if needed
        if use_cache and kv_cache is None:
            kv_cache = [None] * self.num_layers
        
        new_caches = []
        for i, transformer_block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache[i] if kv_cache else None
            x, new_cache = transformer_block(x, kv_cache=layer_cache, use_cache=use_cache, training=training)
            new_caches.append(new_cache)
        
        output = self.dense(x)
        
        if use_cache:
            return output, new_caches
        else:
            return output


def train():
    """Train the transformer model and save it."""
    # Configuration - Change these parameters as needed
    corpus_source = "local"  # Options: "web" for Shakespeare dataset, "local" for corpus.txt
    
    # 1. Load the corpus based on the specified source
    text = load_corpus(corpus_source)
    print("Preview of the dataset:")
    print(text[:500])

    # 2. Preprocess the dataset
    vocab_size = 10000
    seq_length = 100

    # Adapt TextVectorization to full text
    vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int')
    text_ds = tf.data.Dataset.from_tensor_slices([text]).batch(1)
    vectorizer.adapt(text_ds)

    # Vectorize the text
    vectorized_text = vectorizer([text])[0]
    print(f"Vectorized text shape: {vectorized_text.shape}")
    print(f"First 10 vectorized tokens: {vectorized_text.numpy()[:10]}")

    # 3. Generate sequences
    X, Y = create_sequences(vectorized_text.numpy(), seq_length)
    
    # Check if sequences are correctly generated
    print(f"Number of sequences generated: {len(X)}")
    
    # Check if X and Y are not empty
    assert X.size > 0, "Input data X is empty"
    assert Y.size > 0, "Target data Y is empty"
    
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)
    print(f"Shape of X: {X.shape}")
    print(f"Shape of Y: {Y.shape}")

    # 4. Build the Transformer model
    embed_dim = 256
    num_heads = 4
    ff_dim = 512
    num_layers = 4

    model = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length)

    # Provide input shape to build the model
    _ = model(tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()

    # Generate and save model architecture visualization using VisualKeras
    print("Generating model architecture visualization...")
    arch_path = 'transformer_text_model_architecture.png'
    try:
        visualkeras.layered_view(
            model,
            to_file=arch_path,
            legend=True,          # Show layer names
            draw_volume=False,    # Don't show 3D volume representation
            scale_xy=1.5,         # Scale factor for x,y dimensions
            scale_z=1,            # Scale factor for z dimension
            spacing=20            # Spacing between layers
        )
        print(f"Model architecture visualization saved to: {arch_path}")
    except Exception as e:
        print(f"Could not generate VisualKeras visualization: {e}")

    # 5. Setup TensorBoard logging
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    print(f"TensorBoard logs in: {os.path.abspath(logdir)}")
    print(f"Run: tensorboard --logdir {logdir}")

    # Log the VisualKeras image to TensorBoard for easy viewing
    try:
        with tf.summary.create_file_writer(logdir).as_default():
            img = tf.io.read_file(arch_path)
            img = tf.image.decode_png(img, channels=4)
            tf.summary.image("Model Architecture Visualization", tf.expand_dims(img, 0), step=0)
        print("Model visualization logged to TensorBoard")
    except Exception as e:
        print(f"Could not log VisualKeras image to TensorBoard: {e}")

    # 6. Train the model
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    
    print("Starting training...")
    history = model.fit(
        X, Y, 
        epochs=20, 
        batch_size=32, 
        callbacks=[early_stopping, tensorboard_cb]
    )

    print("Training completed!")

    # Save the model weights and architecture parameters separately
    weights_save_path = "transformer_model.weights.h5"
    model.save_weights(weights_save_path)
    print(f"Model weights saved to: {weights_save_path}")

    # Save the vectorizer and model parameters for later use
    import pickle
    vectorizer_path = "text_vectorizer.pkl"
    with open(vectorizer_path, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'vocab_size': vocab_size,
            'seq_length': seq_length,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'num_layers': num_layers
        }, f)
    print(f"Vectorizer and model parameters saved to: {vectorizer_path}")

    # 7. Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    # Save plot to file
    plot_path = os.path.join(logdir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training loss plot saved to: {plot_path}")

    return model, vectorizer, vocab_size, seq_length, logdir


def generate(use_kv_cache=True, weights_path="transformer_model.weights.h5", 
             vectorizer_path="text_vectorizer.pkl"):
    """Load the trained model and generate text."""
    import pickle
    print("Loading trained model and vectorizer...")
    
    # Load the vectorizer and parameters
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer_data = pickle.load(f)
            vectorizer = vectorizer_data['vectorizer']
            vocab_size = vectorizer_data['vocab_size']
            seq_length = vectorizer_data['seq_length']
            embed_dim = vectorizer_data['embed_dim']
            num_heads = vectorizer_data['num_heads']
            ff_dim = vectorizer_data['ff_dim']
            num_layers = vectorizer_data['num_layers']
        print(f"Vectorizer and parameters loaded from: {vectorizer_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}. Please run training first.")
    except KeyError as e:
        raise KeyError(f"Missing parameter in vectorizer file: {e}. Please retrain the model to save all parameters.")
    
    # Reconstruct the model architecture
    print("Reconstructing model architecture...")
    model = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length)
    
    # Build the model by running a forward pass
    dummy_input = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)
    _ = model(dummy_input)
    
    # Load the trained weights
    try:
        model.load_weights(weights_path)
        print(f"Model weights loaded from: {weights_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Weights file not found: {weights_path}. Please run training first.")
    except Exception as e:
        raise Exception(f"Error loading weights: {e}")

    # Create output directory for generated text
    output_dir = os.path.join("generated_outputs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Generate text with the trained model
    print(f"\nGenerating text with KV Cache: {use_kv_cache}")
    start_string = "the object of our"
    
    generated_text = inference(model, vectorizer, start_string, seq_length, 
                                               num_generate=100, temperature=0.7, use_kv_cache=use_kv_cache)
    
    print(f"\nGenerated text: {generated_text}")


def main():
    """Main function to run training and generation."""
    
    # Configuration variables - Change these as needed
    mode = 'generate'  # Options: 'train', 'generate
    use_kv_cache = True  # Set to False to disable KV cache optimization
    weights_path = 'transformer_model.weights.h5'
    vectorizer_path = 'text_vectorizer.pkl'
    
    if mode == 'train':
        print("="*60)
        print("TRAINING MODE")
        print("="*60)
        train()
        
    elif mode == 'generate':
        print("="*60)
        print("GENERATION MODE")
        print("="*60)
        generate(use_kv_cache=use_kv_cache, 
                weights_path=weights_path, 
                vectorizer_path=vectorizer_path)
    

if __name__ == "__main__":
    main()
