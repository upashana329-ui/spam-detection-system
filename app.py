"""
Flask Web Application for Spam Detection System
Same models, same accuracy, same preprocessing - Just web interface!
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Import your existing modules
from data_preprocessing import TextPreprocessor
from model import SpamClassifier

app = Flask(__name__)
app.secret_key = 'spam_detection_secret_key_2024'
CORS(app)

# Global variables
classifier = None
preprocessor = None
is_trained = False

def initialize_and_train():
    """Initialize and train all models - Same as your GUI version"""
    global classifier, preprocessor, is_trained
    
    print("="*60)
    print("🤖 Initializing Spam Detection System")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Load dataset
    texts, labels = preprocessor.load_csv_dataset('data/spam.csv')
    
    if texts is None:
        print("⚠️ CSV not found, using sample dataset")
        texts, labels = preprocessor.load_sample_dataset()
    
    print(f"✅ Loaded {len(texts)} messages")
    print(f"   Spam: {sum(labels)}, Ham: {len(labels)-sum(labels)}")
    
    # Prepare features
    X, y = preprocessor.prepare_data(texts, labels, fit_vectorizer=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Initialize and train models
    classifier = SpamClassifier()
    classifier.initialize_models()
    classifier.train_all_models(X_train, y_train, X_test, y_test)
    
    is_trained = True
    
    # Print performance summary
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE SUMMARY")
    print("="*60)
    for model_name, metrics in classifier.model_performance.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
    
    print("\n" + "="*60)
    print(f"🏆 Best Model: {classifier.current_model_name}")
    print(f"🎯 Accuracy: {classifier.model_performance[classifier.current_model_name]['accuracy']*100:.2f}%")
    print("="*60)
    
    return True

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    if not is_trained:
        return jsonify({
            'trained': False,
            'message': 'System not trained yet'
        })
    
    # Get performance of all models
    performance = {}
    for model_name, metrics in classifier.model_performance.items():
        performance[model_name] = {
            'accuracy': round(metrics['accuracy'] * 100, 2),
            'precision': round(metrics['precision'] * 100, 2),
            'recall': round(metrics['recall'] * 100, 2),
            'f1_score': round(metrics['f1_score'] * 100, 2)
        }
    
    return jsonify({
        'trained': True,
        'current_model': classifier.current_model_name,
        'best_model': classifier.current_model_name,
        'best_accuracy': round(classifier.model_performance[classifier.current_model_name]['accuracy'] * 100, 2),
        'models': performance,
        'dataset_size': len(preprocessor.vectorizer.get_feature_names_out()) if preprocessor else 0
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if a message is spam - Uses your trained model"""
    global classifier, preprocessor
    
    if not is_trained:
        return jsonify({'error': 'Model not trained yet'}), 503
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        model_name = data.get('model', classifier.current_model_name)
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Switch model if requested
        if model_name != classifier.current_model_name:
            classifier.switch_model(model_name)
        
        # Preprocess the message (same as your GUI version)
        processed_text = preprocessor.preprocess(message)
        features = preprocessor.vectorizer.transform([processed_text])
        
        # Predict
        prediction, probability = classifier.predict(features)
        
        # Get prediction details
        is_spam = bool(prediction[0] == 1)
        confidence = float(max(probability[0]) * 100)
        
        # Get feature importance (for spam words highlighting)
        spam_words = []
        if hasattr(classifier.current_model, 'coef_'):
            # Get top spam indicators
            feature_names = preprocessor.vectorizer.get_feature_names_out()
            coef = classifier.current_model.coef_[0]
            top_indices = coef.argsort()[-10:][::-1]
            spam_words = [feature_names[i] for i in top_indices if coef[i] > 0][:5]
        
        return jsonify({
            'is_spam': is_spam,
            'confidence': confidence,
            'model_used': classifier.current_model_name,
            'message': message,
            'spam_indicators': spam_words,
            'probability': {
                'ham': round(probability[0][0] * 100, 2),
                'spam': round(probability[0][1] * 100, 2)
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """Switch between models"""
    global classifier
    
    if not is_trained:
        return jsonify({'error': 'Model not trained yet'}), 503
    
    data = request.get_json()
    model_name = data.get('model')
    
    if classifier.switch_model(model_name):
        return jsonify({
            'success': True,
            'current_model': classifier.current_model_name,
            'accuracy': round(classifier.model_performance[model_name]['accuracy'] * 100, 2)
        })
    else:
        return jsonify({'error': 'Model not found'}), 404

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple messages from CSV"""
    if not is_trained:
        return jsonify({'error': 'Model not trained yet'}), 503
    
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        df = pd.read_csv(file)
        
        # Find text column
        text_col = None
        for col in df.columns:
            if col.lower() in ['text', 'message', 'sms', 'content']:
                text_col = col
                break
        
        if text_col is None:
            text_col = df.columns[0]
        
        results = []
        for idx, row in df.head(100).iterrows():
            message = str(row[text_col])
            processed_text = preprocessor.preprocess(message)
            features = preprocessor.vectorizer.transform([processed_text])
            prediction, probability = classifier.predict(features)
            
            results.append({
                'id': idx,
                'message': message[:100] + '...' if len(message) > 100 else message,
                'is_spam': bool(prediction[0] == 1),
                'confidence': float(max(probability[0]) * 100)
            })
        
        spam_count = sum(1 for r in results if r['is_spam'])
        
        return jsonify({
            'total': len(results),
            'spam_count': spam_count,
            'ham_count': len(results) - spam_count,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain models with new data"""
    global is_trained
    
    try:
        initialize_and_train()
        return jsonify({
            'success': True,
            'message': 'Models retrained successfully',
            'accuracy': round(classifier.model_performance[classifier.current_model_name]['accuracy'] * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize on startup
print("\n🚀 Starting Spam Detection Web Server...")
initialize_and_train()
print("\n✅ Server ready! Visit http://localhost:5000")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)