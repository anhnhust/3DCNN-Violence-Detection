import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class VideoDataset(Dataset):
    """Custom dataset for loading video sequences"""
    
    def __init__(self, data_path, sequence_length=16, img_size=(112, 112), transform=None):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.transform = transform
        
        # Load video paths and labels
        self.video_paths = []
        self.labels = []
        
        # Assuming directory structure: data_path/violence/ and data_path/non_violence/
        violence_path = os.path.join(data_path, 'violence')
        non_violence_path = os.path.join(data_path, 'non_violence')
        
        if os.path.exists(violence_path):
            for video_file in os.listdir(violence_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    self.video_paths.append(os.path.join(violence_path, video_file))
                    self.labels.append(1)  # Violence class
        
        if os.path.exists(non_violence_path):
            for video_file in os.listdir(non_violence_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    self.video_paths.append(os.path.join(non_violence_path, video_file))
                    self.labels.append(0)  # Non-violence class
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        
        if self.transform:
            frames = self.transform(frames)
        
        return torch.FloatTensor(frames), torch.LongTensor([label])
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames"""
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
        
        return frames