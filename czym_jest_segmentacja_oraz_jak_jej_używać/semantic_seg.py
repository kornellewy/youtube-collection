"""
https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
"""

import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import cv2


class SemanticSegmentaion(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.coco_classes_list = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        self.colors = (colors % 255).numpy().astype("uint8")

    def predict_numpy(self, image):
        return self._predict_numpy(image)

    def _predict_numpy(self, image):
        image = Image.fromarray(image)
        image_tensor = self.img_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
        output = torch.argmax(output['out'].squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        for idx, coco_class in enumerate(self.coco_classes_list):
            if idx == 0:
                continue
            output = np.where(output==(idx, idx, idx), self.colors[idx], output).astype(np.uint8)
        return output


if __name__ == "__main__":
    image_path = 'test.jpg'
    kjn = SemanticSegmentaion()
    image = cv2.imread(image_path)
    output = kjn.predict_numpy(image)
    output = cv2.addWeighted(image,0.5,output,0.5,0)
    cv2.imshow("dupa.jpg", output)
    cv2.waitKey(0)
