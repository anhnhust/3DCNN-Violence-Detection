import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
import json
from pathlib import Path

from models import Violence3DCNN
from dataset import VideoDataset

class ViolenceInference:
    """Inference class for violence detection"""
    
    def __init__(self, model_path, device='auto', sequence_length=16, img_size=(112, 112)):
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        # Load model
        self.model = Violence3DCNN(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def preprocess_video(self, video_path):
        """Preprocess video for inference"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        if total_frames > self.sequence_length:
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        else:
            indices = list(range(total_frames))
            # Pad with last frame if needed
            while len(indices) < self.sequence_length:
                indices.append(indices[-1])
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Resize frame
                frame = cv2.resize(frame, self.img_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        
        # Convert to numpy array and transpose to (T, C, H, W)
        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
        
        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)  # (1, T, C, H, W)
        
        return torch.FloatTensor(frames)
    
    def predict(self, video_path):
        """Predict violence in a video"""
        # Preprocess video
        video_tensor = self.preprocess_video(video_path)
        video_tensor = video_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(video_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1)
        
        # Get results
        violence_prob = probabilities[0][1].item()
        non_violence_prob = probabilities[0][0].item()
        is_violent = predicted_class[0].item() == 1
        
        return {
            'is_violent': is_violent,
            'violence_probability': violence_prob,
            'non_violence_probability': non_violence_prob,
            'confidence': max(violence_prob, non_violence_prob)
        }
    
    def predict_batch(self, video_paths):
        """Predict violence for multiple videos"""
        results = []
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                result['video_path'] = video_path
                result['status'] = 'success'
                results.append(result)
                print(f"✓ {video_path}: {'Violence' if result['is_violent'] else 'Non-Violence'} "
                      f"(confidence: {result['confidence']:.3f})")
            except Exception as e:
                results.append({
                    'video_path': video_path,
                    'status': 'error',
                    'error': str(e)
                })
                print(f"✗ {video_path}: Error - {str(e)}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Violence Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--video_path', type=str, help='Path to single video file')
    parser.add_argument('--video_dir', type=str, help='Path to directory containing videos')
    parser.add_argument('--output_path', type=str, default='inference_results.json', help='Output results file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--sequence_length', type=int, default=16, help='Number of frames per sequence')
    parser.add_argument('--img_size', type=tuple, default=(112, 112), help='Image size')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = ViolenceInference(
        model_path=args.model_path,
        device=args.device,
        sequence_length=args.sequence_length,
        img_size=args.img_size
    )
    
    results = []
    
    if args.video_path:
        # Single video inference
        print(f"Processing single video: {args.video_path}")
        result = inference.predict(args.video_path)
        result['video_path'] = args.video_path
        results.append(result)
        
        print(f"\nResult:")
        print(f"Video: {args.video_path}")
        print(f"Prediction: {'Violence' if result['is_violent'] else 'Non-Violence'}")
        print(f"Violence Probability: {result['violence_probability']:.3f}")
        print(f"Non-Violence Probability: {result['non_violence_probability']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    elif args.video_dir:
        # Batch inference
        video_dir = Path(args.video_dir)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_paths = []
        
        for ext in video_extensions:
            video_paths.extend(video_dir.glob(f"*{ext}"))
            video_paths.extend(video_dir.glob(f"*{ext.upper()}"))
        
        if not video_paths:
            print(f"No video files found in {args.video_dir}")
            return
        
        print(f"Processing {len(video_paths)} videos from {args.video_dir}")
        results = inference.predict_batch([str(p) for p in video_paths])
    
    else:
        print("Please provide either --video_path or --video_dir")
        return
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_path}")

if __name__ == "__main__":
    main()