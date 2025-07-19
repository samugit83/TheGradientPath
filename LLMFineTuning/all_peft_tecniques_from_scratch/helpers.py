import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_file


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


def inference(model, vectorizer, start_string, seq_length, num_generate=100, temperature=1.0, use_kv_cache=True):
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
