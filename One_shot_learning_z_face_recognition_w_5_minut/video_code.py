
"""
source:
https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pprint import pprint

import torch
from torchvision import transforms

from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceDetector3(object):
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = MTCNN(keep_all=True, device=self.device, min_face_size=40)

    def detect_from_numpy(self, numpy_img):
        return self._detect_from_numpy(numpy_img)

    def _detect_from_numpy(self, numpy_img):
        image = Image.fromarray(numpy_img)
        boxes, prob = self.model.detect(image)
        prob = prob.tolist()
        boxes = boxes.tolist()
        for idx, bbox in enumerate(boxes):
            new_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])]
            boxes[idx] = new_bbox
        rois = []
        for bbox in boxes:
            roi = numpy_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            rois.append(roi)
        return boxes, prob, rois


class FaceEmbeder(object):
    # img to vec
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def detect_from_numpy(self, numpy_img):
        return self._detect_from_numpy(numpy_img)

    def _detect_from_numpy(self, numpy_img):
        image = Image.fromarray(numpy_img)
        image = transforms.ToTensor()(image).to(self.device).unsqueeze(0)
        embedding = self.model(image).detach().cpu().tolist()[0]
        return embedding
        

class FaceDistanceCalculator(object):
    def calculate_distance(self, vec1, vec2, mode='euclidean'):
        return self._calculate_distance(vec1, vec2, mode)

    def _calculate_distance(self, vec1, vec2, mode):
        if mode == 'euclidean':
            distance = np.linalg.norm(np.array(vec1)-np.array(vec2)).tolist()
        elif mode == 'cosine':
            distance = np.dot(np.array(vec1),np.array(vec2))/\
                (np.linalg.norm(np.array(vec1))*np.linalg.norm(np.array(vec2)))
            distance = np.absolute(distance)
            distance = distance.tolist()
        return distance


class FaceRecognitionSystem(object):
    def __init__(self):
        self.face_detector = FaceDetector3()
        self.face_embeder = FaceEmbeder()
        self.distance_calulator = FaceDistanceCalculator()

        self.distance_treshold = 0.6
        self.face_treshold = 0.85
        self.face_image_size = (100, 100)

        self.database = self._init_database()

    def _init_database(self):
        images_paths = ['img_databese/chege.jpg','img_databese/karol.jpg',
                        'img_databese/lenin.jpg','img_databese/mao.jpg']
        database = {}
        for image_path in images_paths:
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.face_image_size)
            embeding = self.face_embeder.detect_from_numpy(image)
            database.update({os.path.basename(image_path): embeding})
        return database

    def predict_image_numpy(self, image):
        return self._predict_image_numpy(image)

    def _predict_image_numpy(self, image):
        image = cv2.resize(image, self.face_image_size)
        input_face_embeding = self.face_embeder.detect_from_numpy(image)
        output_dict = {}
        for face_name, db_face_embedding in self.database.items():
            distance = self.distance_calulator.calculate_distance(db_face_embedding, input_face_embeding)
            output_dict.update({face_name: distance})
        return output_dict

if __name__ == '__main__':
    test_image = cv2.imread('test.jpg')
    kjn = FaceRecognitionSystem()
    output = kjn.predict_image_numpy(test_image)
    pprint(output)
