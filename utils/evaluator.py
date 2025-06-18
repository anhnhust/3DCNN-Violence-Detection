import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import os

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device).squeeze()
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        report = classification_report(all_targets, all_predictions, 
                                     target_names=['Non-Violence', 'Violence'],
                                     output_dict=True)
        cm = confusion_matrix(all_targets, all_predictions)
        
        return accuracy, report, cm, all_predictions, all_targets, all_probs
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Violence', 'Violence'],
                    yticklabels=['Non-Violence', 'Violence'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_evaluation_results(self, accuracy, report, cm):
        """Print evaluation results"""
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        
        # Format classification report
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"{class_name:12} - Precision: {metrics['precision']:.3f}, "
                      f"Recall: {metrics['recall']:.3f}, F1-score: {metrics['f1-score']:.3f}")
        
        print(f"\nConfusion Matrix:")
        print(f"{'':12} {'Predicted':>20}")
        print(f"{'':12} {'Non-Vio':>8} {'Violence':>10}")
        print(f"{'Actual':12}")
        print(f"{'Non-Vio':12} {cm[0,0]:>8} {cm[0,1]:>10}")
        print(f"{'Violence':12} {cm[1,0]:>8} {cm[1,1]:>10}")