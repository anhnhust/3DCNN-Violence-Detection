import torch
import torch.nn as nn
import torch.nn.functional as F

class Violence3DCNN(nn.Module):
    """3D CNN architecture for violence detection"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(Violence3DCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3d_3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3d_4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, channels, height, width)
        # Transpose to (batch_size, channels, sequence_length, height, width)
        x = x.transpose(1, 2)
        
        # 3D Convolutions
        x = F.relu(self.bn1(self.conv3d_1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv3d_2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3d_3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv3d_4(x)))
        x = self.pool4(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x