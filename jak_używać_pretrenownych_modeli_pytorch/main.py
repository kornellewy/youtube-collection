# importy
import torch
from torchvision import models, transforms
from PIL import Image

if __name__ == "__main__":
    img_path = 'test.jpg'
    # device oraz transformacje
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    # otwarcie oraz konversacja zdjencia
    image = Image.open(img_path)
    image = img_transforms(image)
    image = image.unsqueeze(0)
    # wybranie modelu
    model = models.vgg16(pretrained=True)
    model = model.to(device)
    # uzycie modelu
    with torch.no_grad():
        image = image.to(device)
        pred = model(image)
        pred = torch.exp(pred)
        pred = pred.topk(1, dim=1)
        print(pred)
    
    print(model)
