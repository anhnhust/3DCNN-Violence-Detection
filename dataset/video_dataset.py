import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class VideoDataset(Dataset):
    """Custom dataset for loading video sequences using sliding window approach"""
    
    def __init__(self, data_path, sequence_length=16, img_size=(112, 112), transform=None, 
                 stride=8, min_frames=None):
        """
        Args:
            data_path: Path to dataset directory
            sequence_length: Number of consecutive frames to extract (default: 16)
            img_size: Target image size (height, width)
            transform: Optional transforms to apply
            stride: Step size for sliding window (default: 8, creates overlap)
            min_frames: Minimum frames required in video (default: sequence_length)
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.transform = transform
        self.stride = stride
        self.min_frames = min_frames or sequence_length
        
        # Store video sequences and labels
        self.sequences = []
        self.labels = []
        
        # Load video paths and create sliding window sequences
        self._load_sequences()
    
    def _load_sequences(self):
        """Load video paths and create sliding window sequences"""
        violence_path = os.path.join(self.data_path, 'violence')
        non_violence_path = os.path.join(self.data_path, 'non_violence')
        
        # Process violence videos
        if os.path.exists(violence_path):
            self._process_videos(violence_path, label=1)
        
        # Process non-violence videos
        if os.path.exists(non_violence_path):
            self._process_videos(non_violence_path, label=0)
        
        print(f"Total sequences created: {len(self.sequences)}")
    
    def _process_videos(self, video_dir, label):
        """Process videos in directory and create sliding window sequences"""
        for video_file in os.listdir(video_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(video_dir, video_file)
                
                # Get video info
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Skip videos that are too short
                if total_frames < self.min_frames:
                    print(f"Skipping {video_file}: only {total_frames} frames (minimum: {self.min_frames})")
                    continue
                
                # Create sliding window sequences
                num_sequences = self._calculate_num_sequences(total_frames)
                
                for i in range(num_sequences):
                    start_frame = i * self.stride
                    end_frame = start_frame + self.sequence_length
                    
                    # Ensure we don't exceed video length
                    if end_frame <= total_frames:
                        sequence_info = {
                            'video_path': video_path,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'video_file': video_file
                        }
                        self.sequences.append(sequence_info)
                        self.labels.append(label)
                
                print(f"Created {num_sequences} sequences from {video_file} ({total_frames} frames)")
    
    def _calculate_num_sequences(self, total_frames):
        """Calculate number of sequences that can be extracted from video"""
        if total_frames < self.sequence_length:
            return 0
        
        # Calculate how many sliding windows fit
        return max(1, (total_frames - self.sequence_length) // self.stride + 1)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        label = self.labels[idx]
        
        # Load consecutive frames
        frames = self.load_consecutive_frames(
            sequence_info['video_path'],
            sequence_info['start_frame'],
            sequence_info['end_frame']
        )
        
        if self.transform:
            frames = self.transform(frames)
        
        return torch.FloatTensor(frames), torch.LongTensor([label])
    
    def load_consecutive_frames(self, video_path, start_frame, end_frame):
        """Load consecutive frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read consecutive frames
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                # If we can't read more frames, duplicate the last frame
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    # Create a black frame if no frames were read
                    frame = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                continue
            
            # Preprocess frame
            frame = cv2.resize(frame, self.img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Ensure we have exactly sequence_length frames
        while len(frames) < self.sequence_length:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                # Create black frames if needed
                black_frame = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32)
                frames.append(black_frame)
        
        # Convert to numpy array and transpose to (T, C, H, W)
        frames = np.array(frames[:self.sequence_length])  # Ensure exact length
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
        
        return frames
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        violence_count = sum(1 for label in self.labels if label == 1)
        non_violence_count = sum(1 for label in self.labels if label == 0)
        
        # Get unique videos
        unique_videos = set(seq['video_path'] for seq in self.sequences)
        
        info = {
            'total_sequences': len(self.sequences),
            'violence_sequences': violence_count,
            'non_violence_sequences': non_violence_count,
            'unique_videos': len(unique_videos),
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'img_size': self.img_size
        }
        
        return info