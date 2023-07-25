import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, size_z):
        super(MLP, self).__init__()
        self.size_z = size_z
        self.layers = nn.Sequential(
            nn.Linear(self.size_z , int(self.size_z  / 2)),
            nn.ReLU(True),
            nn.Linear(int(self.size_z  / 2), int(self.size_z  / 4)),
            nn.ReLU(True),
            nn.Linear(int(self.size_z  / 4), 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.layers(input)