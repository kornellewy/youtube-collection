"""
source: 
https://www.researchgate.net/profile/Sheraz-Khan-14/publication/321586653/figure/fig4/AS:568546847014912@1512563539828/The-LeNet-5-Architecture-a-convolutional-neural-network.png
https://www.youtube.com/watch?v=fcOW-Zyb5Bo
https://cs231n.github.io/convolutional-networks/
"""


import torch
import torchvision
import torch.nn as nn

# W2 = (W1-K+2P)/S+1
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2,2))
        # W2 = (W1-K+2P)/S+1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        # W2 = (32+1 -5+2*0)/(1+1) = 14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        # W3 = (14+1 -5+2*0)/(1+1) = 5
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        # W3 = (5+1 -5+2*0)/(1+1) = 5
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        return x

if __name__ == '__main__':
    x = torch.rand(1, 1, 32, 32)
    model = LeNet()
    model(x)
    
