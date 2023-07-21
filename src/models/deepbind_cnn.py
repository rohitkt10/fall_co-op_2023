import torch
import torch.nn as nn

# Define the CNN model
class BasicCNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 16, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(16, output_size)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.mean(x, dim=2)  # Global Average Pooling
        # x = self.dropout(x)  # dropout after pooling
        x = self.fc(x)
        return x
