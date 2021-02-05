import time
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

train_set = datasets.ImageFolder('quantization/test_dataset', img_transform)
classes = train_set.classes

batch_size = 80

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=0)

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(classes)))
model.to(device)

# torch.save(model.state_dict(), 'models/epoch_0_.pth')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(3):
    print("epoch start :", epoch)
    model.train()
    train_loss = 0.0
    start_time = time.time()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logps = model(inputs)
            loss = criterion(logps, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        train_loss += loss.item()
    end_time = time.time()
    epoch_time = end_time - start_time
    train_loss = train_loss/len(train_loader)
    # torch.save(model.state_dict(), 'models/epoch_'+str(epoch+1) +'_.pth')
    print(epoch + 1, train_loss)
    print("epoch_time: ", epoch_time)
