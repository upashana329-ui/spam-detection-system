import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Set style
plt.style.use('dark_background')
sns.set_style("darkgrid")

class MatplotlibCanvas(FigureCanvas):
    """Canvas for embedding matplotlib figures in PyQt5"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#2b2b2b')
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#3c3c3c')
        
    def plot_confusion_matrix(self, cm, classes=['Ham (0)', 'Spam (1)']):
        """Plot confusion matrix with proper formatting"""
        self.axes.clear()
        
        # Convert to integers
        cm_int = cm.astype(int)
        
        # Create heatmap
        sns.heatmap(cm_int, annot=True, fmt='d', cmap='RdYlGn_r', 
                    xticklabels=classes, yticklabels=classes, 
                    ax=self.axes, cbar=True, 
                    annot_kws={'size': 16, 'weight': 'bold'})
        
        self.axes.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', color='white')
        self.axes.set_ylabel('True Label', fontsize=12, fontweight='bold', color='white')
        self.axes.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='white', pad=20)
        
        # Change tick colors
        self.axes.tick_params(colors='white')
        
        self.draw()
        
    def plot_performance_comparison(self, performance_data):
        """Plot model performance comparison"""
        self.axes.clear()
        
        if not performance_data:
            self.axes.text(0.5, 0.5, 'No models trained yet!\nClick "Train Models"',
                          ha='center', va='center', transform=self.axes.transAxes,
                          color='white', fontsize=12)
            self.draw()
            return
        
        # Filter models with reasonable accuracy
        valid_models = {}
        for model_name, metrics in performance_data.items():
            if metrics['accuracy'] > 0.5:
                valid_models[model_name] = metrics
        
        if not valid_models:
            self.axes.text(0.5, 0.5, 'Training in progress...\nPlease wait',
                          ha='center', va='center', transform=self.axes.transAxes,
                          color='white', fontsize=12)
            self.draw()
            return
        
        models = list(valid_models.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        x = np.arange(len(models))
        width = 0.2
        
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            values = [valid_models[model][metric] for model in models]
            bars = self.axes.bar(x + i*width, values, width, 
                                 label=label, 
                                 color=color, 
                                 alpha=0.8,
                                 edgecolor='white',
                                 linewidth=1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    self.axes.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                  f'{height:.2f}', ha='center', va='bottom', 
                                  fontsize=9, color='white', weight='bold')
        
        self.axes.set_xlabel('Models', fontsize=12, fontweight='bold', color='white')
        self.axes.set_ylabel('Scores', fontsize=12, fontweight='bold', color='white')
        self.axes.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', color='white', pad=20)
        self.axes.set_xticks(x + width * 1.5)
        self.axes.set_xticklabels(models, fontsize=10, color='white')
        self.axes.legend(loc='lower right', facecolor='#3c3c3c', labelcolor='white')
        self.axes.set_ylim(0, 1.1)
        self.axes.grid(True, alpha=0.3, axis='y', color='white')
        self.axes.tick_params(colors='white')
        
        # Set spine colors
        for spine in self.axes.spines.values():
            spine.set_color('white')
        
        self.draw()