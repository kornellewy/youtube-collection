import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder('test_dataset', img_transform)
classes = train_set.classes

batch_size = 64

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=0)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(classes)))
model.to(device)

torch.save(model.state_dict(), 'models/epoch_0_.pth')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(10):
    print("epoch start :", epoch)
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss/len(train_loader)
    torch.save(model.state_dict(), 'models/epoch_'+str(epoch+1) +'_.pth')
    print(epoch + 1, train_loss)
