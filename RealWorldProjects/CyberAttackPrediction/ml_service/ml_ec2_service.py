"""
ML EC2 WEB SERVICE - Attack Prediction API
==========================================

This service runs on your EC2 and provides HTTP endpoint for attack prediction.
It loads your trained models and processes network metrics from ALB Lambda.
"""

from flask import Flask, request, jsonify
import numpy as np
import torch 
import logging
import json
from datetime import datetime 

# Import incremental training functionality
from incremental_train import IncrementalTrainer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
ae = None
orc = None
rf = None
preprocessor = None

# Global incremental trainer instance
incremental_trainer = None

def load_training_metadata():
    """Load training metadata to get correct dimensions."""
    try:
        with open("artifacts/training_metadata.json", "r") as f:
            metadata = json.load(f)
        
        processed_features = metadata["training_stats"]["processed_features"]
        selected_features = metadata["training_stats"]["selected_features"]
        
        logger.info(f"Training metadata loaded: {processed_features} processed features, {selected_features} selected features")
        return processed_features, selected_features
        
    except Exception as e:
        logger.warning(f"Could not load training metadata: {e}")
        logger.info("Using default dimensions: 73 processed, 40 selected")
        return 73, 40


def initialize_incremental_trainer():
    """Initialize the global incremental trainer."""
    global incremental_trainer
    try:
        incremental_trainer = IncrementalTrainer("artifacts")
        if incremental_trainer.initialize_or_load_models():
            logger.info("✅ Incremental trainer initialized successfully")
        else:
            logger.warning("⚠️ Incremental trainer initialization failed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize incremental trainer: {e}")

@app.route('/predict', methods=['POST'])
def predict_attack():
    """Predict if network traffic is an attack."""
    global incremental_trainer
    
    try:
        # Get network metrics from Lambda
        raw_data = request.json
        
        logger.info(f"Received prediction request from IP: {raw_data.get('srcip', 'unknown')}")
        
        # Use incremental trainer models for consistency
        if incremental_trainer is None:
            initialize_incremental_trainer()
            
        if incremental_trainer is None or not incremental_trainer.is_initialized:
            return jsonify({'error': 'Models not initialized'}), 500
        
        # Use incremental trainer's models (consistent with streaming training)
        preprocessor = incremental_trainer.preprocessor
        ae = incremental_trainer.ae
        orc = incremental_trainer.orc_sel
        rf = incremental_trainer.rf
        
        # Check if feature selection is enabled from training metadata
        metadata_path = "artifacts/training_metadata.json"
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            apply_feature_selection = metadata.get('apply_feature_selection', True)
        except Exception as e:
            logger.warning(f"Could not load feature selection flag from metadata: {e}")
            logger.info("Using default: feature selection ENABLED")
        
        # Preprocess raw data
        x_processed = preprocessor.transform_single(raw_data)
        
        if apply_feature_selection:
            # FEATURE SELECTION ENABLED: Use AutoEncoder + ORC pipeline
            x_tensor = torch.from_numpy(x_processed.astype(np.float32))
            
            # AutoEncoder reconstruction
            recon = ae.forward_no_grad(x_tensor)
            err = np.abs(x_processed - recon.numpy())
            
            # Update ORC feature selector
            orc.update(err)
            mask_idx = orc.get_mask_indices()
            
            # Create reduced feature set
            feature_names = orc.feature_names
            x_reduced = {feature_names[i]: float(x_processed[i]) for i in mask_idx}
            features_used = [feature_names[i] for i in mask_idx]
        else:
            # FEATURE SELECTION DISABLED: Use all features directly
            feature_names = preprocessor.get_feature_names()
            x_reduced = {feature_names[i]: float(x_processed[i]) for i in range(len(x_processed))}
            features_used = feature_names
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Feature selection DISABLED - using all {len(x_processed)} features")
        
        # Make prediction
        proba = rf.predict_proba(x_reduced)
        attack_prob = proba.get(1, 0.0)
        prediction = 1 if attack_prob >= rf.cfg.attack_threshold else 0
        
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'prediction': prediction,
            'attack_probability': float(attack_prob),
            'source_ip': raw_data.get('srcip', 'unknown'),
            'features_used': features_used,
            'feature_selection_enabled': apply_feature_selection
        }
        
        logger.info(f"Prediction: {prediction} (prob: {attack_prob:.3f}) for IP: {raw_data.get('srcip')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_streaming():
    """Train models incrementally with streaming data."""
    global incremental_trainer
    
    try:

        training_data = request.json
        
        if not training_data or 'flows' not in training_data:
            return jsonify({'error': 'Invalid request format. Expected "flows" field.'}), 400
        
        flows = training_data['flows']
        batch_size = len(flows)
        
        logger.info(f"Received streaming training batch: {batch_size} samples")
        
        if incremental_trainer is None:
            initialize_incremental_trainer()
        
        if incremental_trainer is None:
            return jsonify({'error': 'Failed to initialize incremental trainer'}), 500
        
        result = incremental_trainer.process_streaming_batch(flows)
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown training error')
            logger.error(f"❌ Streaming training failed: {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        logger.info("✅ Training completed - using incremental_trainer models for predictions")
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'batch_size': batch_size,
            'processed_samples': result.get('processed_samples', 0),
            'normal_samples': result.get('normal_samples', 0),
            'attack_samples': result.get('attack_samples', 0),
            'total_processed': result.get('total_processed', 0),
            'batches_processed': result.get('batches_processed', 0),
            'selected_features': result.get('selected_features', 0),
            'preprocessing_updates': result.get('preprocessing_updates', {}),
            'cumulative_class_distribution': {
                'total_normal_samples': result.get('total_normal_samples', 0),
                'total_attack_samples': result.get('total_attack_samples', 0),
                'class_balance_ratio': result.get('class_balance_ratio', 0.0)
            }
        }
        
        logger.info(f"✅ Streaming training completed: {result.get('processed_samples', 0)} samples processed")
        logger.info(f"├─ Batch: Normal: {result.get('normal_samples', 0)}, Attack: {result.get('attack_samples', 0)}")
        logger.info(f"├─ Cumulative: Normal: {result.get('total_normal_samples', 0):,}, Attack: {result.get('total_attack_samples', 0):,}")
        logger.info(f"├─ Class balance ratio: {result.get('class_balance_ratio', 0.0):.2f}:1")
        logger.info(f"├─ Total processed: {result.get('total_processed', 0):,}")
        logger.info(f"└─ Selected features: {result.get('selected_features', 0)}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error in streaming training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/training-status', methods=['GET'])
def get_training_status():
    """Get current incremental training status."""
    global incremental_trainer
    
    try:
        if incremental_trainer is None:
            return jsonify({
                'initialized': False,
                'message': 'Incremental trainer not initialized'
            })
        
        status = incremental_trainer.get_training_status()
        status['timestamp'] = datetime.utcnow().isoformat()
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for ALB."""
    # Check if incremental trainer models are ready
    incremental_models_ready = (
        incremental_trainer is not None and 
        incremental_trainer.is_initialized and
        all([
            incremental_trainer.preprocessor is not None,
            incremental_trainer.ae is not None,
            incremental_trainer.orc_sel is not None,
            incremental_trainer.rf is not None
        ])
    )
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': all([ae, orc, rf, preprocessor]),  # Legacy global models
        'incremental_trainer_ready': incremental_trainer is not None and incremental_trainer.is_initialized,
        'incremental_models_ready': incremental_models_ready,  # The models actually used for prediction/training
        'primary_model_system': 'incremental_trainer'  # Indicate which system is primary
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        'service': 'ML Attack Predictor',
        'status': 'running',
        'version': '1.0',
        'endpoints': ['/health', '/predict', '/train', '/training-status']
    })


try:
    initialize_incremental_trainer()
except Exception as e:
    logger.error(f"❌ Failed to initialize service: {e}")
    logger.info("Service will still start but may not work properly until models are available")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)