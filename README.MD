A PyTorch implementation of violence detection in videos using 3D Convolutional Neural Networks.

## Project Structure

```
violence_detection/
├── data/                   # Dataset directory
│   ├── violence/          # Violence videos
│   └── non_violence/      # Non-violence videos
├── models/                # Model definitions
│   ├── __init__.py
│   └── violence_3dcnn.py  # 3D CNN architecture
├── dataset/               # Dataset utilities
│   ├── __init__.py
│   └── video_dataset.py   # Video dataset loader
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── trainer.py         # Training utilities
│   ├── evaluator.py       # Evaluation utilities
│   └── config.py          # Configuration management
├── checkpoints/           # Saved models
├── results/               # Training results and plots
├── train.py              # Training script
├── test.py               # Testing script
├── inference.py          # Inference script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/anhnhust/3DCNN-Violence-Detection.git
cd 3DCNN-Violence-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Create `data/violence/` and `data/non_violence/` directories
   - Place video files in respective directories
   - Supported formats: .mp4, .avi, .mov, .mkv

## Usage

### Training

Train the model with default parameters:
```bash
python train.py --data_path data/ --epochs 50 --batch_size 8
```

Train with custom parameters:
```bash
python train.py \
    --data_path data/ \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --sequence_length 32 \
    --model_path checkpoints/my_model.pth
```

### Testing

Evaluate the trained model:
```bash
python test.py --model_path checkpoints/best_model.pth --data_path data/
```

### Inference

Single video inference:
```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --video_path path/to/video.mp4
```

Batch inference on directory:
```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --video_dir path/to/videos/ \
    --output_path results.json
```

## Model Architecture

The 3D CNN architecture consists of:
- **Input**: Video sequences (16 frames, 112×112 pixels)
- **3D Convolutions**: 4 layers with increasing channels (64→128→256→512)
- **Batch Normalization**: After each convolution
- **Max Pooling**: Spatial and temporal downsampling
- **Global Average Pooling**: Feature aggregation
- **Classification Head**: Dense layers with dropout

## Configuration Options

Key parameters that can be configured:

- `--data_path`: Path to dataset directory
- `--sequence_length`: Number of frames per video sequence (default: 16)
- `--img_size`: Input image size (default: 112×112)
- `--batch_size`: Training batch size (default: 8)
- `--learning_rate`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 50)
- `--dropout_rate`: Dropout rate (default: 0.5)
- `--device`: Computing device (auto/cuda/cpu)

## Results

After training, you'll find:
- `checkpoints/best_model.pth`: Best model weights
- `results/training_history.png`: Training curves
- `results/confusion_matrix.png`: Test confusion matrix
- `results/test_results.json`: Detailed evaluation metrics

## Model Performance

The model outputs:
- **Binary Classification**: Violence vs Non-Violence
- **Confidence Scores**: Probability for each class
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## Tips for Better Performance

1. **Data Quality**: Ensure balanced dataset with clear violence/non-violence examples
2. **Sequence Length**: Experiment with different frame counts (8, 16, 32)
3. **Data Augmentation**: Add temporal and spatial augmentations
4. **Hyperparameter Tuning**: Adjust learning rate, batch size, dropout
5. **Transfer Learning**: Use pretrained 3D models as backbone

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Low Accuracy**: Check data quality, increase epochs, or adjust learning rate
3. **Video Loading Errors**: Ensure OpenCV can read your video formats
4. **Slow Training**: Use GPU, reduce sequence length, or increase batch size
