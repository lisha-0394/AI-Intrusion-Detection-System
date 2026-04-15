import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

# Global variables
model = None
scaler = None
encoder = None
feature_names = None

def load_models():
    """Load all required models and preprocessors"""
    global model, scaler, encoder, feature_names
    
    try:
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH]):
            raise FileNotFoundError("One or more model files not found!")
        
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        
        print(f"\n{'='*60}")
        print("[SUCCESS] All models loaded successfully!")
        print(f"{'='*60}")
        print(f"[INFO] Model type: {type(model).__name__}")
        print(f"[INFO] Expected features: {len(feature_names)}")
        print(f"[INFO] First 10 features: {feature_names[:10]}")
        print(f"{'='*60}\n")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load models: {str(e)}")
        print(traceback.format_exc())
        return False

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return list of expected features"""
    if feature_names is None:
        return jsonify({'error': 'Models not loaded'}), 500
    return jsonify({'features': feature_names, 'count': len(feature_names)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction based on input features"""
    try:
        if model is None or scaler is None or feature_names is None:
            return jsonify({'error': 'Models not loaded. Please restart the server.'}), 500
        
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Build feature array in correct order
        features_list = []
        missing_features = []
        
        for feature in feature_names:
            if feature in data:
                try:
                    features_list.append(float(data[feature]))
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid value for feature {feature}'}), 400
            else:
                missing_features.append(feature)
        
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {missing_features[:5]}...' if len(missing_features) > 5 else f'Missing required features: {missing_features}'
            }), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features_list).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get confidence
        confidence = float(max(prediction_proba)) * 100
        
        # Determine result
        result = 'ATTACK' if prediction == 1 else 'BENIGN'
        attack_probability = float(prediction_proba[1]) * 100
        benign_probability = float(prediction_proba[0]) * 100
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': f'{confidence:.2f}%',
            'attack_probability': f'{attack_probability:.2f}%',
            'benign_probability': f'{benign_probability:.2f}%',
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = 'healthy' if model is not None else 'unhealthy'
    return jsonify({
        'status': status,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_count': len(feature_names) if feature_names else 0
    })

@app.route('/api/dataset/stats', methods=['GET'])
def dataset_stats():
    """Load and return dataset statistics"""
    try:
        # Load dataset
        dataset_path = r'C:\Users\Admin\Desktop\Sem-6\AI\Project\data\friday.csv'
        df = pd.read_csv(dataset_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Dataset info
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'shape': list(df.shape),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'first_5_rows': df.head(5).to_dict(orient='records'),
            'label_distribution': df['Label'].value_counts().to_dict() if 'Label' in df.columns else {},
            'numerical_stats': df.describe().to_dict()
        }
        
        return jsonify({
            'success': True,
            'dataset': 'friday.csv',
            'stats': stats
        })
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {str(e)}")
        return jsonify({'error': f'Failed to load dataset: {str(e)}'}), 400

@app.route('/api/dataset/sample', methods=['GET'])
def dataset_sample():
    """Return sample rows from dataset"""
    try:
        limit = request.args.get('limit', 10, type=int)
        dataset_path = r'C:\Users\Admin\Desktop\Sem-6\AI\Project\data\friday.csv'
        df = pd.read_csv(dataset_path)
        
        df.columns = df.columns.str.strip()
        sample = df.head(limit)
        
        return jsonify({
            'success': True,
            'count': len(sample),
            'data': sample.to_dict(orient='records')
        })
    except Exception as e:
        print(f"[ERROR] Failed to get sample: {str(e)}")
        return jsonify({'error': f'Failed to get sample: {str(e)}'}), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Make predictions for multiple records"""
    try:
        if model is None or scaler is None or feature_names is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Get CSV data from request
        data = request.json
        records = data.get('records', [])
        
        if not records or len(records) == 0:
            return jsonify({'error': 'No records provided'}), 400
        
        if len(records) > 1000:
            return jsonify({'error': 'Maximum 1000 records per batch'}), 400
        
        results = []
        errors = []
        
        for idx, record in enumerate(records):
            try:
                # Build feature array in correct order
                features_list = []
                missing_features = []
                
                for feature in feature_names:
                    if feature in record:
                        try:
                            value = float(record[feature])
                            features_list.append(value)
                        except (ValueError, TypeError):
                            missing_features.append(feature)
                            features_list.append(0)  # Default to 0 for invalid values
                    else:
                        missing_features.append(feature)
                        features_list.append(0)  # Default to 0 for missing values
                
                # Convert to numpy array and reshape
                features_array = np.array(features_list).reshape(1, -1)
                
                # Scale features
                features_scaled = scaler.transform(features_array)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
                
                # Get confidence
                confidence = float(max(prediction_proba)) * 100
                result = 'ATTACK' if prediction == 1 else 'BENIGN'
                
                results.append({
                    'index': idx,
                    'prediction': result,
                    'attack_probability': float(prediction_proba[1]) * 100,
                    'benign_probability': float(prediction_proba[0]) * 100,
                    'confidence': f'{confidence:.2f}%'
                })
            except Exception as e:
                errors.append({
                    'index': idx,
                    'error': str(e)
                })
        
        # Summary statistics
        attack_count = len([r for r in results if r['prediction'] == 'ATTACK'])
        benign_count = len([r for r in results if r['prediction'] == 'BENIGN'])
        
        return jsonify({
            'success': True,
            'total_records': len(records),
            'processed': len(results),
            'errors': len(errors),
            'attack_count': attack_count,
            'benign_count': benign_count,
            'results': results,
            'error_details': errors[:10]  # Show first 10 errors
        })
        
    except Exception as e:
        print(f"[ERROR] Batch prediction failed: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models before starting server
    if load_models():
        print("[INFO] Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("[ERROR] Could not start server. Models failed to load.")