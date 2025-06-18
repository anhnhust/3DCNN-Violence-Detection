import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

from models import Violence3DCNN
from dataset import VideoDataset
from utils import ModelEvaluator, get_config

def main():
    # Get configuration
    config = get_config()
    
    # Device configuration
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.FloatTensor(x))
    ])
    
    # Create dataset
    print("Loading dataset...")
    dataset = VideoDataset(
        data_path=config.data_path,
        sequence_length=config.sequence_length,
        img_size=config.img_size,
        transform=transform
    )
    
    # Split dataset (use same splits as training)
    dataset_size = len(dataset)
    train_size = int(config.train_split * dataset_size)
    val_size = int(config.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    _, _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    model = Violence3DCNN(
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate
    )
    
    # Load trained model
    print(f"Loading model from: {config.model_path}")
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, config)
    
    # Evaluate model
    print("Evaluating model on test set...")
    accuracy, report, cm, predictions, targets, probs = evaluator.evaluate(test_loader)
    
    # Print results
    evaluator.print_evaluation_results(accuracy, report, cm)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        cm, 
        save_path=f"{config.results_dir}/confusion_matrix.png"
    )
    
    # Save detailed results
    results = {
        'test_accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions,
        'targets': targets
    }
    
    with open(f"{config.results_dir}/test_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {config.results_dir}/test_results.json")

if __name__ == "__main__":
    main()