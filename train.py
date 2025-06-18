import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np

from models import Violence3DCNN
from dataset import VideoDataset
from utils import ViolenceDetectionTrainer, get_config, save_config

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    # Get configuration
    config = get_config()
    
    # Set random seed
    set_seed(42)
    
    # Device configuration
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {vars(config)}")
    
    # Save configuration
    save_config(config, f"{config.results_dir}/train_config.json")
    
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
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(config.train_split * dataset_size)
    val_size = int(config.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = Violence3DCNN(
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params} total parameters ({trainable_params} trainable)")
    
    # Create trainer
    trainer = ViolenceDetectionTrainer(model, device, config)
    
    # Train model
    print("Starting training...")
    best_val_acc = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history()
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.model_path}")

if __name__ == "__main__":
    main()