import sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from data_preprocessing import TextPreprocessor
from model import SpamClassifier
from utils import MatplotlibCanvas
from sklearn.model_selection import train_test_split

class SpamDetectionGUI(QMainWindow):
    """Main GUI Application for Spam Detection System"""
    
    def __init__(self):
        super().__init__()
        self.preprocessor = TextPreprocessor()
        self.classifier = SpamClassifier()
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🤖 AI-Powered Spam Detection System")
        self.setGeometry(100, 100, 1400, 850)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 13px;
            }
            QTextEdit, QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 2px solid #3c3c3c;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
                font-family: 'Segoe UI', Arial;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 8px;
                min-width: 180px;
                font-size: 13px;
                font-family: 'Segoe UI', Arial;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QGroupBox {
                color: #4CAF50;
                border: 2px solid #3c3c3c;
                border-radius: 10px;
                margin-top: 15px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
            }
            QTextEdit {
                font-family: 'Segoe UI', Arial;
            }
            QStatusBar {
                color: white;
                background-color: #2d2d2d;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("🛡️ AI-Powered Spam Detection System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            margin: 15px; 
            color: #4CAF50;
            font-family: 'Segoe UI', Arial;
        """)
        main_layout.addWidget(title_label)
        
        # Create split layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        
        # Left panel - Input section
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # Text input group
        input_group = QGroupBox("📝 Message Input")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type or paste your message here...")
        self.text_input.setMaximumHeight(200)
        self.text_input.setFont(QFont("Segoe UI", 12))
        input_layout.addWidget(self.text_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.check_btn = QPushButton("🔍 Check Spam")
        self.check_btn.setEnabled(False)
        self.clear_btn = QPushButton("🗑️ Clear")
        self.upload_btn = QPushButton("📁 Upload CSV")
        button_layout.addWidget(self.check_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.upload_btn)
        input_layout.addLayout(button_layout)
        
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # Result display group
        result_group = QGroupBox("🎯 Detection Result")
        result_layout = QVBoxLayout()
        result_layout.setSpacing(10)
        
        self.result_label = QLabel("⚡ No message analyzed yet")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            padding: 20px;
            background-color: #2d2d2d;
            border-radius: 10px;
        """)
        result_layout.addWidget(self.result_label)
        
        self.prob_label = QLabel("📊 Confidence: --%")
        self.prob_label.setAlignment(Qt.AlignCenter)
        self.prob_label.setStyleSheet("font-size: 14px; padding: 5px;")
        result_layout.addWidget(self.prob_label)
        
        result_group.setLayout(result_layout)
        left_layout.addWidget(result_group)
        
        # Model selection group
        model_group = QGroupBox("🧠 Model Selection & Training")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Logistic Regression', 'Naive Bayes', 'Random Forest'])
        self.model_combo.setEnabled(False)
        model_layout.addWidget(self.model_combo)
        
        self.train_btn = QPushButton("🎯 Train All Models")
        self.train_btn.setStyleSheet("background-color: #2196F3;")
        model_layout.addWidget(self.train_btn)
        
        self.save_btn = QPushButton("💾 Save Model")
        self.save_btn.setEnabled(False)
        self.load_btn = QPushButton("📂 Load Model")
        self.load_btn.setEnabled(False)
        
        save_load_layout = QHBoxLayout()
        save_load_layout.addWidget(self.save_btn)
        save_load_layout.addWidget(self.load_btn)
        model_layout.addLayout(save_load_layout)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Statistics group
        stats_group = QGroupBox("📊 Model Performance Metrics")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)
        
        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Confusion matrix
        cm_group = QGroupBox("📈 Confusion Matrix")
        cm_layout = QVBoxLayout()
        self.canvas_cm = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        cm_layout.addWidget(self.canvas_cm)
        cm_group.setLayout(cm_layout)
        right_layout.addWidget(cm_group)
        
        # Performance comparison
        perf_group = QGroupBox("📊 Performance Comparison")
        perf_layout = QVBoxLayout()
        self.canvas_perf = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        perf_layout.addWidget(self.canvas_perf)
        perf_group.setLayout(perf_layout)
        right_layout.addWidget(perf_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([550, 750])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("✅ Ready - Click 'Train All Models' to begin")
        self.statusBar.setStyleSheet("QStatusBar { background-color: #2d2d2d; color: white; padding: 5px; }")
        
        # Load data on startup
        self.load_sample_data()
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.check_btn.clicked.connect(self.check_spam)
        self.clear_btn.clicked.connect(self.clear_text)
        self.upload_btn.clicked.connect(self.upload_csv)
        self.train_btn.clicked.connect(self.train_models)
        self.model_combo.currentTextChanged.connect(self.switch_model)
        self.save_btn.clicked.connect(self.save_model)
        self.load_btn.clicked.connect(self.load_model)
        
    def load_sample_data(self):
        """Load data from CSV or sample"""
        try:
            # Try to load from CSV first
            print("\n" + "="*60)
            print("📂 LOADING DATASET")
            print("="*60)
            
            texts, labels = self.preprocessor.load_csv_dataset('data/spam.csv')
            
            if texts is not None and labels is not None and len(texts) > 0:
                self.sample_texts = texts
                self.sample_labels = labels
                spam_count = sum(labels)
                ham_count = len(labels) - spam_count
                
                status_msg = f"✅ Loaded {len(texts)} messages from CSV ({spam_count} spam, {ham_count} ham)"
                self.statusBar.showMessage(status_msg)
                print(f"\n{status_msg}")
                
                # Show warning if dataset is small or imbalanced
                if len(texts) < 50:
                    print(f"⚠️ Warning: Small dataset ({len(texts)} messages). Add more data for better accuracy.")
                if spam_count < 5:
                    print(f"⚠️ Warning: Very few spam messages ({spam_count}). Add more spam examples.")
                    
            else:
                # Fallback to sample
                print("\n⚠️ No valid CSV found, using sample dataset...")
                texts, labels = self.preprocessor.load_sample_dataset()
                self.sample_texts = texts
                self.sample_labels = labels
                self.statusBar.showMessage(f"⚠️ Using sample dataset ({len(texts)} messages)")
                print(f"✅ Using sample dataset: {len(texts)} messages")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Emergency fallback
            self.sample_texts = ["Test message", "Spam message"]
            self.sample_labels = [0, 1]
            self.statusBar.showMessage("⚠️ Using minimal dataset")
            
    def train_models(self):
        """Train all machine learning models"""
        try:
            self.train_btn.setEnabled(False)
            self.train_btn.setText("⏳ Training in Progress...")
            self.statusBar.showMessage("🔄 Training models... Please wait...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Prepare data
            X, y = self.preprocessor.prepare_data(self.sample_texts, self.sample_labels, fit_vectorizer=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            print(f"\n📊 Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
            print(f"📊 Training - Spam: {sum(y_train)}, Ham: {len(y_train)-sum(y_train)}")
            
            # Initialize and train models
            self.classifier.initialize_models()
            self.classifier.train_all_models(X_train, y_train, X_test, y_test)
            
            # Update UI
            self.update_statistics()
            current_model = self.model_combo.currentText()
            self.update_visualizations(current_model)
            
            # Enable controls
            self.check_btn.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            
            self.statusBar.showMessage("✅ All models trained successfully!")
            QApplication.restoreOverrideCursor()
            
            # Show success message
            best_model = self.classifier.current_model_name
            best_acc = self.classifier.model_performance[best_model]['accuracy'] * 100
            
            QMessageBox.information(self, "Training Complete", 
                f"✅ Models trained successfully!\n\n"
                f"🏆 Best Model: {best_model}\n"
                f"📊 Accuracy: {best_acc:.2f}%\n\n"
                f"💡 You can now test messages or upload CSV files!")
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"❌ Training failed: {str(e)}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.train_btn.setEnabled(True)
            self.train_btn.setText("🎯 Train All Models")
            
    def update_statistics(self):
        """Update model performance statistics display"""
        performance = self.classifier.get_model_performance()
        
        stats_text = "<b>📊 Model Performance Summary</b><br><br>"
        
        for model_name, metrics in performance.items():
            stats_text += f"<b>🔹 {model_name}</b><br>"
            stats_text += f"   • Accuracy: <b style='color: #4CAF50;'>{metrics['accuracy']*100:.2f}%</b><br>"
            stats_text += f"   • Precision: {metrics['precision']*100:.2f}%<br>"
            stats_text += f"   • Recall: {metrics['recall']*100:.2f}%<br>"
            stats_text += f"   • F1-Score: {metrics['f1_score']*100:.2f}%<br><br>"
        
        # Highlight current model
        current_model = self.model_combo.currentText()
        if current_model in performance:
            stats_text += f"<b>✅ Currently using: {current_model}</b><br>"
            stats_text += f"   Accuracy: <b style='color: #4CAF50;'>{performance[current_model]['accuracy']*100:.2f}%</b>"
        
        self.stats_text.setHtml(stats_text)
        
    def update_visualizations(self, model_name):
        """Update confusion matrix and performance charts"""
        performance = self.classifier.get_model_performance()
        
        if model_name in performance:
            # Update confusion matrix
            cm = performance[model_name]['confusion_matrix']
            self.canvas_cm.plot_confusion_matrix(cm)
            
            # Update performance comparison
            self.canvas_perf.plot_performance_comparison(performance)
            
    def switch_model(self):
        """Switch between different models"""
        if self.classifier.trained:
            model_name = self.model_combo.currentText()
            if self.classifier.switch_model(model_name):
                self.update_visualizations(model_name)
                self.statusBar.showMessage(f"🔄 Switched to {model_name} model")
                self.update_statistics()
                
    def check_spam(self):
        """Check if the input text is spam"""
        if not self.classifier.trained:
            QMessageBox.warning(self, "Warning", "⚠️ Please train the models first!")
            return
            
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "⚠️ Please enter a message to analyze!")
            return
            
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)
            features = self.preprocessor.vectorizer.transform([processed_text])
            
            # Predict
            prediction, probability = self.classifier.predict(features)
            
            # Update result display
            if prediction[0] == 1:
                self.result_label.setText("🚨 SPAM DETECTED! 🚨")
                self.result_label.setStyleSheet("""
                    font-size: 22px; 
                    font-weight: bold; 
                    padding: 20px;
                    background-color: #ff4444;
                    border-radius: 10px;
                    color: white;
                """)
                self.prob_label.setText(f"⚠️ Confidence: {probability[0][1]*100:.2f}% that this is SPAM")
                self.prob_label.setStyleSheet("color: #ff8888; font-size: 14px;")
            else:
                self.result_label.setText("✅ NOT SPAM (HAM) ✅")
                self.result_label.setStyleSheet("""
                    font-size: 22px; 
                    font-weight: bold; 
                    padding: 20px;
                    background-color: #4CAF50;
                    border-radius: 10px;
                    color: white;
                """)
                self.prob_label.setText(f"🔒 Confidence: {probability[0][0]*100:.2f}% that this is HAM")
                self.prob_label.setStyleSheet("color: #88ff88; font-size: 14px;")
                
            self.statusBar.showMessage("✅ Analysis complete!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Prediction failed: {str(e)}")
            
    def clear_text(self):
        """Clear the text input"""
        self.text_input.clear()
        self.result_label.setText("⚡ No message analyzed yet")
        self.result_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            padding: 20px;
            background-color: #2d2d2d;
            border-radius: 10px;
            color: white;
        """)
        self.prob_label.setText("📊 Confidence: --%")
        self.prob_label.setStyleSheet("color: white; font-size: 14px;")
        
    def upload_csv(self):
        """Upload and process CSV file"""
        if not self.classifier.trained:
            QMessageBox.warning(self, "Warning", "⚠️ Please train the models first!")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                
                # Auto-detect text column
                text_col = None
                for col in df.columns:
                    if col.lower() in ['text', 'message', 'sms', 'content']:
                        text_col = col
                        break
                
                if text_col is None and len(df.columns) >= 1:
                    text_col = df.columns[0]
                
                if text_col:
                    texts = df[text_col].astype(str).tolist()
                    
                    # Process each text (limit to 100 for performance)
                    results = []
                    for text in texts[:100]:
                        processed_text = self.preprocessor.preprocess(str(text))
                        features = self.preprocessor.vectorizer.transform([processed_text])
                        prediction, probability = self.classifier.predict(features)
                        results.append({
                            'text': text[:50] + "..." if len(str(text)) > 50 else text,
                            'prediction': 'SPAM' if prediction[0] == 1 else 'HAM',
                            'confidence': probability[0][prediction[0]] * 100
                        })
                    
                    # Show results
                    self.show_results_dialog(results)
                    self.statusBar.showMessage(f"✅ Processed {len(results)} messages from CSV")
                    
                else:
                    QMessageBox.warning(self, "Error", "❌ Could not find text column in CSV")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"❌ Failed to process CSV: {str(e)}")
                
    def show_results_dialog(self, results):
        """Show batch processing results in a dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("📊 Batch Processing Results")
        dialog.setGeometry(200, 200, 900, 600)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"📈 Processed {len(results)} Messages")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create table
        table = QTableWidget()
        table.setRowCount(len(results))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Message", "Prediction", "Confidence"])
        table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                color: white;
                gridline-color: #3c3c3c;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-size: 13px;
            }
        """)
        
        spam_count = 0
        for i, result in enumerate(results):
            table.setItem(i, 0, QTableWidgetItem(str(result['text'])))
            
            pred_item = QTableWidgetItem(result['prediction'])
            if result['prediction'] == 'SPAM':
                pred_item.setForeground(QColor(255, 68, 68))
                spam_count += 1
            else:
                pred_item.setForeground(QColor(76, 175, 80))
            table.setItem(i, 1, pred_item)
            
            conf_item = QTableWidgetItem(f"{result['confidence']:.2f}%")
            table.setItem(i, 2, conf_item)
        
        table.resizeColumnsToContents()
        table.setAlternatingRowColors(True)
        layout.addWidget(table)
        
        # Summary
        summary = QLabel(f"📊 Summary: {spam_count} SPAM | {len(results)-spam_count} HAM")
        summary.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px; color: #4CAF50;")
        summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(summary)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                padding: 10px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def save_model(self):
        """Save trained model to file"""
        if self.classifier.trained:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "spam_model.pkl", "Pickle Files (*.pkl)")
            if file_path:
                if self.classifier.save_model(file_path):
                    QMessageBox.information(self, "Success", f"✅ Model saved to {file_path}")
                else:
                    QMessageBox.warning(self, "Error", "❌ Failed to save model")
                    
    def load_model(self):
        """Load trained model from file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Pickle Files (*.pkl)")
        if file_path:
            if self.classifier.load_model(file_path):
                self.classifier.trained = True
                self.update_statistics()
                self.update_visualizations(self.classifier.current_model_name)
                self.check_btn.setEnabled(True)
                self.model_combo.setEnabled(True)
                self.model_combo.setCurrentText(self.classifier.current_model_name)
                self.statusBar.showMessage(f"✅ Model loaded from {file_path}")
                QMessageBox.information(self, "Success", "✅ Model loaded successfully!")
            else:
                QMessageBox.warning(self, "Error", "❌ Failed to load model")