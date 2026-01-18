"""
Model Evaluation Metrics
Implements accuracy, precision, recall, F1-score, and confusion matrix
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    
    def __init__(self, labels: List[str] = ['positive', 'negative', 'neutral']):
        """
        Initialize evaluator
        
        Args:
            labels: List of class labels
        """
        self.labels = labels
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, labels=self.labels, 
                                                      average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, labels=self.labels, 
                                               average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, labels=self.labels, 
                                      average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, labels=self.labels, 
                                             average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=self.labels, 
                                       average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=self.labels, 
                               average=None, zero_division=0)
        
        for i, label in enumerate(self.labels):
            metrics[f'precision_{label}'] = precision_per_class[i]
            metrics[f'recall_{label}'] = recall_per_class[i]
            metrics[f'f1_{label}'] = f1_per_class[i]
        
        # Weighted metrics
        metrics['precision_weighted'] = precision_score(y_true, y_pred, labels=self.labels, 
                                                        average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, labels=self.labels, 
                                                 average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, labels=self.labels, 
                                         average='weighted', zero_division=0)
        
        return metrics
    
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as numpy array
        """
        return confusion_matrix(y_true, y_pred, labels=self.labels)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for title
            save_path: Path to save the plot (optional)
        """
        cm = self.confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.labels, yticklabels=self.labels)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        return classification_report(y_true, y_pred, labels=self.labels, 
                                    target_names=self.labels, zero_division=0)
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model", plot_cm: bool = True,
                      save_path: Optional[str] = None) -> Dict:
        """
        Complete evaluation pipeline
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            plot_cm: Whether to plot confusion matrix
            save_path: Path to save plots (optional)
            
        Returns:
            Dictionary with all metrics and evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}\n")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Print metrics
        print("Overall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (Macro):  {metrics['f1_macro']:.4f}")
        print(f"\n  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (Weighted):    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")
        
        print("\nPer-Class Metrics:")
        for label in self.labels:
            print(f"\n  {label.capitalize()}:")
            print(f"    Precision: {metrics[f'precision_{label}']:.4f}")
            print(f"    Recall:    {metrics[f'recall_{label}']:.4f}")
            print(f"    F1-Score:  {metrics[f'f1_{label}']:.4f}")
        
        # Confusion matrix
        cm = self.confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print("\nClassification Report:")
        print(self.classification_report(y_true, y_pred))
        
        # Plot confusion matrix
        if plot_cm:
            cm_path = save_path.replace('.png', '_cm.png') if save_path else None
            self.plot_confusion_matrix(y_true, y_pred, model_name=model_name, save_path=cm_path)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': self.classification_report(y_true, y_pred)
        }
    
    def compare_models(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                      save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results: Dictionary with model_name: (y_true, y_pred) pairs
            save_path: Path to save comparison plot (optional)
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison = []
        
        for model_name, (y_true, y_pred) in results.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['model'] = model_name
            comparison.append(metrics)
        
        df_comparison = pd.DataFrame(comparison)
        
        # Reorder columns
        cols = ['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
               'precision_weighted', 'recall_weighted', 'f1_weighted']
        df_comparison = df_comparison[cols + [c for c in df_comparison.columns if c not in cols]]
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(df_comparison.to_string(index=False))
        
        # Plot comparison
        if save_path:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx // 2, idx % 2]
                df_comparison.plot(x='model', y=metric, kind='bar', ax=ax, legend=False)
                ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison plot saved to {save_path}")
            plt.show()
        
        return df_comparison


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Sample data
    y_true = np.array(['positive', 'negative', 'neutral', 'positive', 'negative'] * 20)
    y_pred = np.array(['positive', 'negative', 'positive', 'positive', 'negative'] * 20)
    
    results = evaluator.evaluate_model(y_true, y_pred, model_name="Sample Model")
    print("\nMetrics:", results['metrics'])
