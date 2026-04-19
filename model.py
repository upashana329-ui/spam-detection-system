import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
import joblib
import os

class SpamClassifier:
    """Multi-model spam classifier system with proper training"""
    
    def __init__(self):
        self.models = {}
        self.current_model_name = None
        self.current_model = None
        self.trained = False
        self.model_performance = {}
        
    def initialize_models(self):
        """Initialize models with proper parameters"""
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=0.5),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced',
                C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced',
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            )
        }
        self.current_model_name = 'Logistic Regression'
        self.current_model = self.models[self.current_model_name]
    
    def balance_dataset(self, X, y):
        """Balance the dataset using upsampling"""
        # Convert sparse matrix to array if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Separate classes
        X_ham = X[y == 0]
        y_ham = y[y == 0]
        X_spam = X[y == 1]
        y_spam = y[y == 1]
        
        print(f"   Original - Ham: {len(X_ham)}, Spam: {len(X_spam)}")
        
        if len(X_spam) == 0:
            print("   Warning: No spam samples found!")
            return X, y
        
        # Upsample spam to match ham count
        if len(X_spam) < len(X_ham):
            X_spam_upsampled, y_spam_upsampled = resample(
                X_spam, y_spam,
                replace=True,
                n_samples=len(X_ham),
                random_state=42
            )
            X_balanced = np.vstack((X_ham, X_spam_upsampled))
            y_balanced = np.hstack((y_ham, y_spam_upsampled))
        else:
            # Downsample ham to match spam
            X_ham_downsampled, y_ham_downsampled = resample(
                X_ham, y_ham,
                replace=False,
                n_samples=len(X_spam),
                random_state=42
            )
            X_balanced = np.vstack((X_ham_downsampled, X_spam))
            y_balanced = np.hstack((y_ham_downsampled, y_spam))
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        print(f"   Balanced - Ham: {sum(y_balanced==0)}, Spam: {sum(y_balanced==1)}")
        
        return X_balanced, y_balanced
    
    def train_model(self, model_name, X_train, y_train, X_test, y_test):
        """Train a specific model and evaluate performance"""
        if not self.models:
            self.initialize_models()
        
        # Get the model
        model = self.models[model_name]
        
        # Balance the training data
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train)
        
        print(f"\n🔧 Training {model_name}...")
        
        # Train the model
        model.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store performance
        self.model_performance[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'model': model
        }
        
        print(f"✅ {model_name} - Accuracy: {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%")
        
        return self.model_performance[model_name]
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all available models"""
        if not self.models:
            self.initialize_models()
        
        print("\n" + "="*60)
        print("🎯 Starting model training...")
        print("="*60)
        
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train, X_test, y_test)
        
        self.trained = True
        
        # Set best model as current
        best_model = max(self.model_performance, key=lambda x: self.model_performance[x]['accuracy'])
        self.current_model_name = best_model
        self.current_model = self.model_performance[best_model]['model']
        
        print("\n" + "="*60)
        print(f"🏆 Best model: {best_model} (Accuracy: {self.model_performance[best_model]['accuracy']*100:.2f}%)")
        print("="*60 + "\n")
        
    def switch_model(self, model_name):
        """Switch to a different trained model"""
        if model_name in self.model_performance:
            self.current_model_name = model_name
            self.current_model = self.model_performance[model_name]['model']
            return True
        return False
    
    def predict(self, features):
        """Predict using current model"""
        if not self.trained:
            raise ValueError("Model not trained yet!")
        
        prediction = self.current_model.predict(features)
        probability = self.current_model.predict_proba(features)
        
        return prediction, probability
    
    def get_model_performance(self, model_name=None):
        """Get performance metrics for a model"""
        if model_name:
            return self.model_performance.get(model_name, None)
        return self.model_performance
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        if self.trained:
            model_data = {
                'models': self.models,
                'current_model_name': self.current_model_name,
                'model_performance': self.model_performance,
                'trained': self.trained
            }
            joblib.dump(model_data, filepath)
            return True
        return False
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.current_model_name = model_data['current_model_name']
            self.model_performance = model_data['model_performance']
            self.trained = model_data['trained']
            self.current_model = self.models[self.current_model_name]
            return True
        return False