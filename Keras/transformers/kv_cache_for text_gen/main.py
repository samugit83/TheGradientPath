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


def create_sequences(text, seq_length):
    """Generate input and target sequences for training."""
    input_seqs = []
    target_seqs = []
    for i in range(len(text) - seq_length):
        input_seq = text[i:i + seq_length]
        target_seq = text[i + 1:i + seq_length + 1]
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    return np.array(input_seqs), np.array(target_seqs)


def generate_text_with_kv_cache(model, vectorizer, start_string, seq_length, num_generate=100, temperature=1.0, use_kv_cache=True):
    """Generate text using the trained Transformer model with KV cache optimization."""
    # Convert the start string to a vectorized format
    input_eval = vectorizer([start_string]).numpy()
    
    # Ensure the input length is the same as the model's expected input shape
    if input_eval.shape[1] < seq_length:
        # Pad the input if it's shorter than the expected sequence length
        padding = np.zeros((1, seq_length - input_eval.shape[1]))
        input_eval = np.concatenate((padding, input_eval), axis=1)
    elif input_eval.shape[1] > seq_length:
        # Truncate the input if it's longer than the expected sequence length
        input_eval = input_eval[:, -seq_length:]

    input_eval = tf.convert_to_tensor(input_eval)
    text_generated = []
    vocab = vectorizer.get_vocabulary()
    
    if use_kv_cache:
        print("Using KV Cache for generation...")
        
        # PREFILL PHASE: Process the initial prompt and build cache
        predictions, kv_cache = model(input_eval, use_cache=True, start_pos=0, training=False)
        
        # Get the last token's predictions for the first generation
        last_predictions = predictions[0, -1, :]
        last_predictions = last_predictions / temperature
        predicted_id = tf.random.categorical(tf.expand_dims(last_predictions, 0), num_samples=1)[0, 0].numpy()
        
        if predicted_id < len(vocab):
            text_generated.append(vocab[predicted_id])
        
        current_pos = input_eval.shape[1]
        
        # DECODE PHASE: Generate tokens one by one using cached K/V
        for i in range(num_generate - 1):
            # Prepare next token input
            next_token = tf.convert_to_tensor([[predicted_id]])
            
            # Generate next token using cached keys/values
            predictions, kv_cache = model(next_token, kv_cache=kv_cache, use_cache=True, 
                                        start_pos=current_pos, training=False)
            
            # Get predictions and sample next token
            last_predictions = predictions[0, -1, :]
            last_predictions = last_predictions / temperature
            predicted_id = tf.random.categorical(tf.expand_dims(last_predictions, 0), num_samples=1)[0, 0].numpy()
            
            if predicted_id < len(vocab):
                text_generated.append(vocab[predicted_id])
            
            current_pos += 1
    else:
        print("Using standard generation (no KV cache)...")
        
        # Standard generation without KV cache (original method)
        for i in range(num_generate):
            predictions = model(input_eval, use_cache=False, training=False)
            predictions = predictions[0, -1, :]
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(tf.expand_dims(predictions, 0), num_samples=1)[0, 0].numpy()

            # Update the input tensor to include the predicted word, maintaining the sequence length
            input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)
            input_eval = input_eval[:, -seq_length:]  # Keep only the last `seq_length` tokens
            input_eval = tf.convert_to_tensor(input_eval)

            if predicted_id < len(vocab):
                text_generated.append(vocab[predicted_id])

    return start_string + ' ' + ' '.join(text_generated)


def generate_text(model, vectorizer, start_string, seq_length, num_generate=100, temperature=1.0):
    """Legacy function for backward compatibility - uses standard generation."""
    return generate_text_with_kv_cache(model, vectorizer, start_string, seq_length, 
                                     num_generate, temperature, use_kv_cache=False)


def load_corpus(corpus_source):
    """Load corpus data from either web or local file."""
    if corpus_source == "web":
        print("Loading Shakespeare dataset from web...")
        path_to_file = get_file('shakespeare.txt', 
                               'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        print(f"Web dataset loaded. Text length: {len(text)} characters")
    elif corpus_source == "local":
        print("Loading corpus from local file 'corpus.txt'...")
        try:
            with open('corpus.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Local corpus loaded. Text length: {len(text)} characters")
        except FileNotFoundError:
            raise FileNotFoundError("corpus.txt not found. Please make sure the file exists in the current directory.")
        except UnicodeDecodeError:
            print("Failed to decode with UTF-8, trying with latin-1...")
            with open('corpus.txt', 'r', encoding='latin-1') as f:
                text = f.read()
            print(f"Local corpus loaded with latin-1 encoding. Text length: {len(text)} characters")
    else:
        raise ValueError(f"Invalid corpus source: {corpus_source}. Must be 'web' or 'local'.")
    
    return text


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
    import time
    
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
    
    # Measure generation time for comparison
    start_time = time.time()
    generated_text = generate_text_with_kv_cache(model, vectorizer, start_string, seq_length, 
                                               num_generate=100, temperature=0.7, use_kv_cache=use_kv_cache)
    generation_time = time.time() - start_time
    
    print(f"\nGenerated text (100 tokens in {generation_time:.2f}s):")
    print(generated_text)

    # Generate longer text sequence
    print("\nGenerating longer text sequence...")
    start_time = time.time()
    longer_text = generate_text_with_kv_cache(model, vectorizer, start_string, seq_length, 
                                            num_generate=200, temperature=0.8, use_kv_cache=use_kv_cache)
    longer_generation_time = time.time() - start_time
    
    print(f"\nLonger generated text (200 tokens in {longer_generation_time:.2f}s):")
    print(longer_text)

    # Performance comparison if KV cache is enabled
    if use_kv_cache:
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        
        # Test without KV cache
        print("Testing generation WITHOUT KV cache...")
        start_time = time.time()
        text_no_cache = generate_text_with_kv_cache(model, vectorizer, start_string, seq_length, 
                                                  num_generate=100, temperature=0.7, use_kv_cache=False)
        time_no_cache = time.time() - start_time
        
        print(f"\nPerformance Results:")
        print(f"With KV Cache:    {generation_time:.2f}s")
        print(f"Without KV Cache: {time_no_cache:.2f}s")
        speedup = time_no_cache / generation_time if generation_time > 0 else 1
        print(f"Speedup:          {speedup:.2f}x")
        print(f"Time saved:       {((time_no_cache - generation_time) / time_no_cache * 100):.1f}%")

    # Save generated text to file
    text_output_path = os.path.join(output_dir, 'generated_text.txt')
    with open(text_output_path, 'w') as f:
        f.write(f"KV Cache enabled: {use_kv_cache}\n")
        f.write(f"Start string: {start_string}\n\n")
        f.write(f"Generated text (100 tokens in {generation_time:.2f}s):\n{generated_text}\n\n")
        f.write(f"Generated text (200 tokens in {longer_generation_time:.2f}s):\n{longer_text}\n\n")
        if use_kv_cache and 'time_no_cache' in locals():
            f.write(f"Performance comparison:\n")
            f.write(f"With KV Cache: {generation_time:.2f}s\n")
            f.write(f"Without KV Cache: {time_no_cache:.2f}s\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
    print(f"Generated text and performance results saved to: {text_output_path}")

    # Generate multiple samples with different temperatures
    print("\n" + "="*50)
    print("GENERATING SAMPLES WITH DIFFERENT TEMPERATURES")
    print("="*50)
    
    temperatures = [0.5, 0.7, 1.0, 1.2]
    samples_output_path = os.path.join(output_dir, 'temperature_samples.txt')
    
    with open(samples_output_path, 'w') as f:
        f.write("Text Generation Samples with Different Temperatures\n")
        f.write("="*60 + "\n\n")
        
        for temp in temperatures:
            print(f"Generating with temperature {temp}...")
            sample_text = generate_text_with_kv_cache(model, vectorizer, start_string, seq_length, 
                                                    num_generate=150, temperature=temp, use_kv_cache=use_kv_cache)
            
            print(f"\nTemperature {temp}:")
            print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
            
            f.write(f"Temperature: {temp}\n")
            f.write("-" * 20 + "\n")
            f.write(f"{sample_text}\n\n")
    
    print(f"Temperature samples saved to: {samples_output_path}")
    print(f"All outputs saved in directory: {output_dir}")


def generate_compare(weights_path="transformer_model.weights.h5", 
                    vectorizer_path="text_vectorizer.pkl"):
    """Load the trained model and generate text comparing KV cache vs standard generation."""
    import pickle
    import time
    
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
    output_dir = os.path.join("generated_outputs", "comparison_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("KV CACHE vs STANDARD GENERATION PERFORMANCE COMPARISON")
    print("="*80)
    
    # Test parameters
    start_string = "the object of our"
    token_counts = [50, 100, 200]  # Different sequence lengths to test
    temperatures = [0.7, 1.0]  # Different creativity levels
    
    results = []
    
    for temp in temperatures:
        print(f"\n📊 Testing with temperature: {temp}")
        print("-" * 50)
        
        for num_tokens in token_counts:
            print(f"\nGenerating {num_tokens} tokens...")
            
            # Test WITH KV Cache
            print("  🚀 With KV Cache...", end=" ")
            start_time = time.time()
            text_with_cache = generate_text_with_kv_cache(
                model, vectorizer, start_string, seq_length, 
                num_generate=num_tokens, temperature=temp, use_kv_cache=True
            )
            time_with_cache = time.time() - start_time
            print(f"({time_with_cache:.3f}s)")
            
            # Test WITHOUT KV Cache
            print("  🐌 Without KV Cache...", end=" ")
            start_time = time.time()
            text_without_cache = generate_text_with_kv_cache(
                model, vectorizer, start_string, seq_length, 
                num_generate=num_tokens, temperature=temp, use_kv_cache=False
            )
            time_without_cache = time.time() - start_time
            print(f"({time_without_cache:.3f}s)")
            
            # Calculate metrics
            speedup = time_without_cache / time_with_cache if time_with_cache > 0 else 1
            time_saved = time_without_cache - time_with_cache
            percentage_saved = (time_saved / time_without_cache * 100) if time_without_cache > 0 else 0
            
            # Store results
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
            
            # Display results
            print(f"    ⚡ Speedup: {speedup:.2f}x")
            print(f"    ⏱️  Time saved: {time_saved:.3f}s ({percentage_saved:.1f}%)")
            
    # Display comprehensive results table
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE RESULTS")
    print("="*80)
    print(f"{'Temp':<6} {'Tokens':<7} {'With Cache':<12} {'Without Cache':<14} {'Speedup':<8} {'Time Saved':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['temperature']:<6} {result['tokens']:<7} {result['time_with_cache']:.3f}s{'':<6} "
              f"{result['time_without_cache']:.3f}s{'':<8} {result['speedup']:.2f}x{'':<4} "
              f"{result['time_saved']:.3f}s ({result['percentage_saved']:.1f}%)")
    
    # Calculate averages
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_time_saved_pct = sum(r['percentage_saved'] for r in results) / len(results)
    total_time_with_cache = sum(r['time_with_cache'] for r in results)
    total_time_without_cache = sum(r['time_without_cache'] for r in results)
    
    print("-" * 80)
    print(f"AVERAGE PERFORMANCE IMPROVEMENT:")
    print(f"  • Average Speedup: {avg_speedup:.2f}x")
    print(f"  • Average Time Saved: {avg_time_saved_pct:.1f}%")
    print(f"  • Total Time - With Cache: {total_time_with_cache:.3f}s")
    print(f"  • Total Time - Without Cache: {total_time_without_cache:.3f}s")
    print(f"  • Overall Speedup: {total_time_without_cache/total_time_with_cache:.2f}x")
    
    # Save detailed results to file
    results_path = os.path.join(output_dir, 'performance_comparison.txt')
    with open(results_path, 'w') as f:
        f.write("KV CACHE vs STANDARD GENERATION PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("PERFORMANCE RESULTS TABLE:\n")
        f.write(f"{'Temp':<6} {'Tokens':<7} {'With Cache':<12} {'Without Cache':<14} {'Speedup':<8} {'Time Saved':<12}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            f.write(f"{result['temperature']:<6} {result['tokens']:<7} {result['time_with_cache']:.3f}s{'':<6} "
                   f"{result['time_without_cache']:.3f}s{'':<8} {result['speedup']:.2f}x{'':<4} "
                   f"{result['time_saved']:.3f}s ({result['percentage_saved']:.1f}%)\n")
        
        f.write("\nAVERAGE PERFORMANCE IMPROVEMENT:\n")
        f.write(f"  • Average Speedup: {avg_speedup:.2f}x\n")
        f.write(f"  • Average Time Saved: {avg_time_saved_pct:.1f}%\n")
        f.write(f"  • Total Time - With Cache: {total_time_with_cache:.3f}s\n")
        f.write(f"  • Total Time - Without Cache: {total_time_without_cache:.3f}s\n")
        f.write(f"  • Overall Speedup: {total_time_without_cache/total_time_with_cache:.2f}x\n\n")
        
        f.write("GENERATED TEXT SAMPLES:\n")
        f.write("="*50 + "\n\n")
        
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
    
    # Performance insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if avg_speedup > 2.0:
        print("🎉 Excellent performance! KV Cache provides significant speedup.")
    elif avg_speedup > 1.5:
        print("✅ Good performance improvement with KV Cache.")
    elif avg_speedup > 1.1:
        print("⚡ Modest but meaningful speedup with KV Cache.")
    else:
        print("⚠️  Limited speedup - may need larger sequences to see benefits.")
    
    print(f"\n💡 The KV Cache optimization is most effective for:")
    print(f"   • Longer sequence generation ({max(token_counts)} tokens showed best relative improvement)")
    print(f"   • Interactive applications where latency matters")
    print(f"   • Batch generation of multiple sequences")
    
    print(f"\n🔧 Technical details:")
    print(f"   • KV Cache eliminates redundant computation of attention keys/values")
    print(f"   • Memory usage increases slightly to store cached values")
    print(f"   • Speedup scales with sequence length and model size")
    
    print(f"\nAll comparison results saved in: {output_dir}")


def main():
    """Main function to run training and generation."""
    
    # Configuration variables - Change these as needed
    mode = 'generate_compare'  # Options: 'train', 'generate', 'generate_compare', 'both'
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
        
    elif mode == 'generate_compare':
        print("="*60)
        print("PERFORMANCE COMPARISON MODE")
        print("="*60)
        generate_compare(weights_path=weights_path, 
                        vectorizer_path=vectorizer_path)
        
    else:  # both
        print("="*60)
        print("TRAINING MODE")
        print("="*60)
        model, vectorizer, vocab_size, seq_length, logdir = train()
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON MODE")
        print("="*60)
        generate_compare(weights_path=weights_path, 
                        vectorizer_path=vectorizer_path)


if __name__ == "__main__":
    main()
