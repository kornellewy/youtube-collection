"""
https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
"""

import torch
import random
from PIL import Image
from torchvision import models, transforms
import numpy as np
import cv2


class InstanceSegmentaion(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.model = model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, num_classes=91)
        self.model.to(self.device)
        self.model.eval()
        self.coco_classes_list = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(91)])[:, None] * palette
        self.colors = (colors % 255).numpy().astype("uint8")

    def predict_numpy(self, image):
        return self._predict_numpy(image)

    def _predict_numpy(self, image):
        image = Image.fromarray(image)
        image_tensor = self.img_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        masks, boxes, labels = self._get_outputs(image_tensor)
        output = self._draw_segmentation_map(image, masks, boxes, labels)
        return output

    def _get_outputs(self, image, threshold=0.95):
        with torch.no_grad():
            outputs = self.model(image)
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        masks = masks[:thresholded_preds_count]
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
        boxes = boxes[:thresholded_preds_count]
        labels = [self.coco_classes_list[i] for i in outputs[0]['labels']]
        return masks, boxes, labels

    def _draw_segmentation_map(self, image, masks, boxes, labels):
        alpha = 1 
        beta = 0.6
        gamma = 0
        for i in range(len(masks)):
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            color = self.colors[random.randrange(0, len(self.colors))]
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=tuple(color.tolist()), 
                        thickness=2)
            cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(color.tolist()), 
                        thickness=2, lineType=cv2.LINE_AA)
        return image


if __name__ == "__main__":
    image_path = 'test.jpg'
    kjn = InstanceSegmentaion()
    image = cv2.imread(image_path)
    output = kjn.predict_numpy(image)
    cv2.imshow("dupa.jpg", output)
    cv2.waitKey(0)
