#!/usr/bin/env python3
"""
Complete script to implement Transformers for text generation.
Build, train, and evaluate an LLM using a Transformer model for text generation using TensorFlow and Keras.
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


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        super(TransformerModel, self).__init__()
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

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        x += self.pos_encoding[:, :seq_len, :]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        output = self.dense(x)
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


def generate_text(model, vectorizer, start_string, seq_length, num_generate=100, temperature=1.0):
    """Generate text using the trained Transformer model."""
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
    
    # Initialize an empty list to store generated text
    text_generated = []

    # Start generating text
    for i in range(num_generate):
        # Make predictions using the model
        predictions = model(input_eval)

        # Get the last token's predictions
        predictions = predictions[0, -1, :]

        # Apply temperature to predictions
        predictions = predictions / temperature
        
        # Use a categorical distribution to predict the next word
        predicted_id = tf.random.categorical(tf.expand_dims(predictions, 0), num_samples=1)[0, 0].numpy()

        # Update the input tensor to include the predicted word, maintaining the sequence length
        input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)
        input_eval = input_eval[:, -seq_length:]  # Keep only the last `seq_length` tokens
        input_eval = tf.convert_to_tensor(input_eval)

        # Append the predicted word to the generated text
        vocab = vectorizer.get_vocabulary()
        if predicted_id < len(vocab):
            text_generated.append(vocab[predicted_id])

    # Return the generated text starting from the initial seed
    return start_string + ' ' + ' '.join(text_generated)


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


def main():
    # Configuration - Change this parameter to switch between corpus sources
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

    # Save the trained model
    model_save_path = "transformer_text_generation_model.h5"
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

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

    # 8. Generate text with the trained model
    print("\nGenerating text...")
    start_string = "the object of our"
    generated_text = generate_text(model, vectorizer, start_string, seq_length, 
                                 num_generate=100, temperature=0.7)
    print("\nGenerated text:")
    print(generated_text)

    # 9. Generate longer text sequence
    print("\nGenerating longer text sequence...")
    longer_text = generate_text(model, vectorizer, start_string, seq_length, 
                              num_generate=200, temperature=0.8)
    print("\nLonger generated text:")
    print(longer_text)

    # Save generated text to file
    text_output_path = os.path.join(logdir, 'generated_text.txt')
    with open(text_output_path, 'w') as f:
        f.write(f"Start string: {start_string}\n\n")
        f.write(f"Generated text (100 tokens):\n{generated_text}\n\n")
        f.write(f"Generated text (200 tokens):\n{longer_text}\n")
    print(f"Generated text saved to: {text_output_path}")


if __name__ == "__main__":
    main()
