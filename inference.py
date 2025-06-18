import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Union

from models import Violence3DCNN
from dataset import VideoDataset

class ViolenceInference:
    """Inference class for violence detection with sliding window support"""
    
    def __init__(self, model_path, device='auto', sequence_length=16, img_size=(112, 112), 
                 inference_mode='sliding_window', stride=8, threshold=0.5):
        """
        Args:
            model_path: Path to trained model
            device: Device to use ('cuda', 'cpu', or 'auto')
            sequence_length: Number of frames per sequence
            img_size: Target image size (height, width)
            inference_mode: 'sliding_window', 'uniform_sampling', or 'full_video'
            stride: Stride for sliding window (only used in sliding_window mode)
            threshold: Threshold for violence classification
        """
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.inference_mode = inference_mode
        self.stride = stride
        self.threshold = threshold
        
        # Load model
        self.model = Violence3DCNN(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Inference mode: {self.inference_mode}")
        if self.inference_mode == 'sliding_window':
            print(f"Sliding window stride: {self.stride}")
    
    def extract_sequences_sliding_window(self, video_path):
        """Extract sequences using sliding window approach"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.sequence_length:
            cap.release()
            raise ValueError(f"Video too short: {total_frames} frames (minimum: {self.sequence_length})")
        
        # Calculate number of sequences
        num_sequences = max(1, (total_frames - self.sequence_length) // self.stride + 1)
        sequences = []
        
        for i in range(num_sequences):
            start_frame = i * self.stride
            end_frame = start_frame + self.sequence_length
            
            if end_frame > total_frames:
                break
                
            # Extract consecutive frames
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(self.sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame = cv2.resize(frame, self.img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            if len(frames) == self.sequence_length:
                # Convert to tensor format (T, C, H, W)
                frames = np.array(frames)
                frames = np.transpose(frames, (0, 3, 1, 2))
                sequences.append(frames)
        
        cap.release()
        return sequences
    
    def extract_sequences_uniform(self, video_path):
        """Extract single sequence using uniform sampling (original method)"""
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
        frames = np.transpose(frames, (0, 3, 1, 2))
        
        return [frames]  # Return as list for consistency
    
    def extract_sequences_full_video(self, video_path):
        """Extract overlapping sequences covering the entire video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.sequence_length:
            cap.release()
            raise ValueError(f"Video too short: {total_frames} frames (minimum: {self.sequence_length})")
        
        # Use smaller stride for more comprehensive coverage
        stride = max(1, self.stride // 2)
        num_sequences = max(1, (total_frames - self.sequence_length) // stride + 1)
        sequences = []
        
        for i in range(num_sequences):
            start_frame = i * stride
            end_frame = start_frame + self.sequence_length
            
            if end_frame > total_frames:
                start_frame = total_frames - self.sequence_length
                end_frame = total_frames
            
            # Extract consecutive frames
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(self.sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame = cv2.resize(frame, self.img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            if len(frames) == self.sequence_length:
                frames = np.array(frames)
                frames = np.transpose(frames, (0, 3, 1, 2))
                sequences.append(frames)
        
        cap.release()
        return sequences
    
    def predict_sequences(self, sequences):
        """Predict violence for multiple sequences"""
        predictions = []
        
        for sequence in sequences:
            # Add batch dimension
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(sequence_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                violence_prob = probabilities[0][1].item()
                non_violence_prob = probabilities[0][0].item()
                
                predictions.append({
                    'violence_prob': violence_prob,
                    'non_violence_prob': non_violence_prob,
                    'is_violent': violence_prob > self.threshold
                })
        
        return predictions
    
    def aggregate_predictions(self, predictions):
        """Aggregate predictions from multiple sequences"""
        if not predictions:
            return None
        
        # Calculate average probabilities
        avg_violence_prob = np.mean([p['violence_prob'] for p in predictions])
        avg_non_violence_prob = np.mean([p['non_violence_prob'] for p in predictions])
        
        # Count violent sequences
        violent_sequences = sum(1 for p in predictions if p['is_violent'])
        total_sequences = len(predictions)
        violence_ratio = violent_sequences / total_sequences
        
        # Different aggregation strategies
        strategies = {
            'average': avg_violence_prob > self.threshold,
            'majority': violence_ratio > 0.5,
            'any': violent_sequences > 0,
            'strict': violence_ratio > 0.8  # Require 80% of sequences to be violent
        }
        
        return {
            'is_violent': strategies['majority'],  # Default strategy
            'violence_probability': avg_violence_prob,
            'non_violence_probability': avg_non_violence_prob,
            'confidence': max(avg_violence_prob, avg_non_violence_prob),
            'sequences_analyzed': total_sequences,
            'violent_sequences': violent_sequences,
            'violence_ratio': violence_ratio,
            'strategies': strategies,
            'detailed_predictions': predictions
        }
    
    def predict(self, video_path):
        """Predict violence in a video"""
        try:
            # Extract sequences based on inference mode
            if self.inference_mode == 'sliding_window':
                sequences = self.extract_sequences_sliding_window(video_path)
            elif self.inference_mode == 'uniform_sampling':
                sequences = self.extract_sequences_uniform(video_path)
            elif self.inference_mode == 'full_video':
                sequences = self.extract_sequences_full_video(video_path)
            else:
                raise ValueError(f"Unknown inference mode: {self.inference_mode}")
            
            if not sequences:
                raise ValueError("No sequences extracted from video")
            
            # Get predictions for all sequences
            predictions = self.predict_sequences(sequences)
            
            # Aggregate predictions
            if len(predictions) == 1:
                # Single sequence - return direct result
                result = predictions[0]
                result.update({
                    'sequences_analyzed': 1,
                    'violent_sequences': 1 if result['is_violent'] else 0,
                    'violence_ratio': 1.0 if result['is_violent'] else 0.0,
                    'confidence': max(result['violence_prob'], result['non_violence_prob'])
                })
            else:
                # Multiple sequences - aggregate results
                result = self.aggregate_predictions(predictions)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'is_violent': False,
                'violence_probability': 0.0,
                'non_violence_probability': 1.0,
                'confidence': 0.0
            }
    
    def predict_batch(self, video_paths):
        """Predict violence for multiple videos"""
        results = []
        for i, video_path in enumerate(video_paths, 1):
            print(f"Processing video {i}/{len(video_paths)}: {Path(video_path).name}")
            
            result = self.predict(video_path)
            result['video_path'] = video_path
            
            if 'error' not in result:
                result['status'] = 'success'
                print(f"  ✓ {'Violence' if result['is_violent'] else 'Non-Violence'} "
                      f"(confidence: {result['confidence']:.3f}, "
                      f"sequences: {result.get('sequences_analyzed', 1)})")
            else:
                result['status'] = 'error'
                print(f"  ✗ Error: {result['error']}")
            
            results.append(result)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Violence Detection Inference with Sliding Window')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--video_path', type=str, help='Path to single video file')
    parser.add_argument('--video_dir', type=str, help='Path to directory containing videos')
    parser.add_argument('--output_path', type=str, default='inference_results.json', help='Output results file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--sequence_length', type=int, default=16, help='Number of frames per sequence')
    parser.add_argument('--img_size', type=int, nargs=2, default=[112, 112], help='Image size (height width)')
    parser.add_argument('--inference_mode', type=str, default='sliding_window', 
                       choices=['sliding_window', 'uniform_sampling', 'full_video'],
                       help='Inference mode')
    parser.add_argument('--stride', type=int, default=8, help='Stride for sliding window')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = ViolenceInference(
        model_path=args.model_path,
        device=args.device,
        sequence_length=args.sequence_length,
        img_size=tuple(args.img_size),
        inference_mode=args.inference_mode,
        stride=args.stride,
        threshold=args.threshold
    )
    
    results = []
    
    if args.video_path:
        # Single video inference
        print(f"Processing single video: {args.video_path}")
        result = inference.predict(args.video_path)
        result['video_path'] = args.video_path
        results.append(result)
        
        print(f"\nResults:")
        print(f"Video: {args.video_path}")
        if 'error' not in result:
            print(f"Prediction: {'Violence' if result['is_violent'] else 'Non-Violence'}")
            print(f"Violence Probability: {result['violence_probability']:.3f}")
            print(f"Non-Violence Probability: {result['non_violence_probability']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Sequences Analyzed: {result.get('sequences_analyzed', 1)}")
            if 'violence_ratio' in result:
                print(f"Violence Ratio: {result['violence_ratio']:.3f}")
        else:
            print(f"Error: {result['error']}")
    
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
        
        # Print summary
        successful = [r for r in results if r['status'] == 'success']
        violent_videos = [r for r in successful if r['is_violent']]
        
        print(f"\nSummary:")
        print(f"Total videos: {len(results)}")
        print(f"Successfully processed: {len(successful)}")
        print(f"Violent videos detected: {len(violent_videos)}")
        
    else:
        print("Please provide either --video_path or --video_dir")
        return
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_path}")

if __name__ == "__main__":
    main()