# https://www.youtube.com/watch?v=ACmuBbuXn20&feature=youtu.be
# https://arxiv.org/abs/1409.1556
# https://www.youtube.com/watch?v=Nf1BlOR8kVY

import torch
import torchvision
import torch.nn as nn

# dlaczego powstała
# - bo zauwazo ze gdy zwiększamy sieć jej dokłądność rośnie 
# - sprawdzono że lepiej jest mieć głeboka siec niż szeroką
# - podzas zwększania ilości warstw pojawił sie problem z problamatycznym pisaniem całosci, wiec zastosowano bloki

# 3x3 kernel jest z stride = 1 i pading =1 

# architektura
# int 64 - to znaczy output channels
# 'M - max pull 
VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, type = VGG_types['VGG16']):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layer = self.create_conv_layers(type)

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
        
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                    nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))] 
        return nn.Sequential(*layers)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1,3,224,224).to(device)
    model = VGG(in_channels=3, num_classes=1000, type = VGG_types['VGG11']).to(device)
    print(model)
    print(model(x).shape) #1 batch siez i 1000 prawdopodobiens na kazda klase
