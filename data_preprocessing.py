import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    """Handles all text preprocessing operations"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Keep important negative words
        keep_words = {'not', 'no', 'nor', 'but', 'however', 'although'}
        self.stop_words = self.stop_words - keep_words
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_words(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text, apply_stemming=True):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming
        if apply_stemming:
            tokens = self.stem_words(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def prepare_data(self, texts, labels=None, fit_vectorizer=True):
        """Prepare data for model training/prediction"""
        # Preprocess all texts
        processed_texts = [self.preprocess(text) for text in texts]
        
        if fit_vectorizer:
            # Fit and transform for training
            features = self.vectorizer.fit_transform(processed_texts)
        else:
            # Transform only for prediction
            features = self.vectorizer.transform(processed_texts)
        
        if labels is not None:
            return features, np.array(labels)
        
        return features
    
    def load_csv_dataset(self, file_path='data/spam.csv'):
        """Load dataset from CSV file - Auto-detects format"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                return None, None
            
            print(f"📂 Loading CSV from: {file_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ Loaded with encoding: {encoding}")
                    break
                except:
                    continue
            
            if df is None:
                print("❌ Could not load CSV with any encoding")
                return None, None
            
            print(f"📊 CSV Columns found: {df.columns.tolist()}")
            print(f"📊 Dataset shape: {df.shape}")
            
            # 🔥 AUTO-DETECT COLUMN FORMAT
            text_col = None
            label_col = None
            
            # Check for common column names
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['text', 'message', 'sms', 'email', 'content', 'v2', 'sms_text', 'message_text']:
                    text_col = col
                elif col_lower in ['label', 'class', 'type', 'spam', 'v1', 'category', 'target']:
                    label_col = col
            
            # If not found, try first two columns (common pattern)
            if text_col is None and len(df.columns) >= 2:
                text_col = df.columns[1]  # Second column usually text
                print(f"🔍 Auto-detected text column: {text_col}")
            
            if label_col is None and len(df.columns) >= 1:
                label_col = df.columns[0]  # First column usually label
                print(f"🔍 Auto-detected label column: {label_col}")
            
            if text_col is None or label_col is None:
                print(f"❌ Could not identify text/label columns")
                print(f"   Available columns: {df.columns.tolist()}")
                return None, None
            
            print(f"✅ Using: text='{text_col}', label='{label_col}'")
            
            # Extract data
            texts = df[text_col].astype(str).tolist()
            labels_raw = df[label_col].tolist()
            
            # Convert labels to 0/1
            labels = []
            for label in labels_raw:
                label_str = str(label).lower().strip()
                if label_str in ['spam', '1', 'true', 'yes', 'spam', 'spam']:
                    labels.append(1)
                elif label_str in ['ham', '0', 'false', 'no', 'ham', 'non-spam']:
                    labels.append(0)
                else:
                    # Try to convert to int
                    try:
                        val = int(float(label_str))
                        labels.append(val if val in [0,1] else 0)
                    except:
                        print(f"⚠️ Unknown label: {label}, setting to 0 (ham)")
                        labels.append(0)
            
            spam_count = sum(labels)
            ham_count = len(labels) - spam_count
            
            print(f"✅ Successfully loaded {len(texts)} messages")
            print(f"   📧 Ham (normal): {ham_count} messages")
            print(f"   🚨 Spam: {spam_count} messages")
            
            if spam_count == 0:
                print(f"⚠️ WARNING: No spam messages found! Add spam examples for better training.")
            
            return texts, np.array(labels)
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_sample_dataset(self):
        """Load from CSV first, fallback to sample data"""
        # First try to load from CSV
        print("\n🔍 Looking for CSV file at 'data/spam.csv'...")
        texts, labels = self.load_csv_dataset('data/spam.csv')
        
        if texts is not None and labels is not None and len(texts) > 0:
            print("✅ Using CSV dataset from data/spam.csv")
            return texts, labels
        
        # Fallback to enhanced sample dataset if CSV not found
        print("\n⚠️ CSV not found or invalid, using built-in sample dataset")
        print("💡 Tip: Place your spam.csv file in the 'data' folder")
        
        data = {
            'text': [
                # HAM messages (0)
                "Hey, how are you doing today?",
                "Can we meet for coffee tomorrow?",
                "Don't forget about the meeting at 3pm",
                "Please review the attached document",
                "Your order has been shipped",
                "Thanks for your purchase",
                "What time is the party tonight?",
                "I love spending time with family",
                "The weather is beautiful today",
                "Let's catch up over lunch sometime",
                "Your appointment is confirmed for Monday",
                "Please call me back when you get this",
                "The project deadline is next Friday",
                "I really enjoyed our conversation",
                "Happy birthday! Hope you have a great day",
                "Can you send me the report?",
                "I'll be there in 10 minutes",
                "Did you see the game last night?",
                "What's for dinner today?",
                "I need help with my homework",
                
                # SPAM messages (1)
                "Congratulations! You've won a free iPhone! Click here to claim now!",
                "URGENT: Your account has been compromised. Verify now to secure it!",
                "FREE MONEY!!! Click this link to get $1000 instantly!",
                "You have been selected for a $500 Amazon gift card! Claim now!",
                "WINNER! You are our lucky winner! Call now to claim your prize!",
                "Your credit card has been charged $999. Call immediately to dispute!",
                "Limited time offer! Buy now and get 90% off!",
                "You've won a luxury vacation package! Click here to book!",
                "Earn $5000 per month working from home! No experience needed!",
                "Your PayPal account is suspended! Verify your information now!",
                "Claim your free gift card before it expires!",
                "You have 1 unread message from the lottery department!",
                "Investment opportunity! Double your money in 30 days!",
                "Your Netflix subscription has expired! Update payment now!",
                "Congratulations! You've been pre-approved for a credit card!",
                "Click here to claim your prize!",
                "Last chance to win $1000!",
                "Your account will be closed today!",
                "FREE trial for 30 days!",
                "You've been selected for a special offer!"
            ],
            'label': [0]*20 + [1]*20  # 20 ham, 20 spam (balanced)
        }
        
        print(f"✅ Using sample dataset: {len(data['text'])} messages (20 spam, 20 ham)")
        return data['text'], data['label']