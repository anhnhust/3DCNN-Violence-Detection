import argparse
import json
import os

def get_config():
    """Get configuration from command line arguments and config file"""
    parser = argparse.ArgumentParser(description='Violence Detection using 3D CNN')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/', help='Path to dataset')
    parser.add_argument('--sequence_length', type=int, default=16, help='Number of frames per sequence')
    parser.add_argument('--img_size', type=tuple, default=(112, 112), help='Image size')
    parser.add_argument('--stride', type=int, default=16, help='Stride for sliding window')
    parser.add_argument('--min_frames', type=int, default=16, help='Minimum frames required in video')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to save/load model')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    # Data split
    parser.add_argument('--train_split', type=float, default=0.8, help='Training data split ratio')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation data split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test data split ratio')
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='results/', help='Results directory')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    return args

def save_config(config, path):
    """Save configuration to JSON file"""
    with open(path, 'w') as f:
        json.dump(vars(config), f, indent=2)

def load_config(path):
    """Load configuration from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)