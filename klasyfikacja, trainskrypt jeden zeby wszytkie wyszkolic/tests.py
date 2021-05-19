"""
source:
https://stackoverflow.com/questions/23195522/opencv-fastest-method-to-check-if-two-images-are-100-same-or-not
https://discuss.pytorch.org/t/any-way-to-get-model-name/12877/4
"""

import os
import unittest
import torch
import cv2
from pathlib import Path
from torchvision import datasets, models, transforms
import numpy as np
from pytorch_lightning import Trainer
import json

from code_video import AlbumentationsTransform, ImbalancedDatasetSampler, ClassificationTrainer


class TestAlbumentationsTransform(unittest.TestCase):
    def setUp(self):
        self.test_image_path = 'cat_vs_dog_test_dataset/Dog/1.jpg'

    def test_agumentations(self):
        image = cv2.imread(self.test_image_path)
        test_class = AlbumentationsTransform()
        agu_image = test_class(image)
        self.assertEqual(image.all()==None, False)
        self.assertEqual(agu_image['image'].all() == None, False)
        self.assertEqual(self.is_similar(agu_image['image'], image), False)

    def is_similar(self, image1, image2):
        return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())


class TestImbalancedDatasetSampler(unittest.TestCase):
    def setUp(self):
        self.dataset_path = 'cat_vs_dog_test_dataset'
        self.img_transforms = transforms.Compose([AlbumentationsTransform()])
        self.dataset = datasets.ImageFolder(self.dataset_path, self.img_transforms)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, sampler=ImbalancedDatasetSampler(self.dataset), batch_size=32)

    def test_batch_class_balance(self):
        """
        test_dataset count: 339
        cat: 248 73%
        dog: 91 27%
        """
        class_0_count = 0
        class_1_count = 0
        for i in range(10):
            for images, labels in self.dataloader:
                for label in labels:
                    if label == 0:
                        class_0_count+=1
                    else:
                        class_1_count+=1
        sum_count = class_0_count + class_1_count
        unqi_class_weights = [round(i, 3) for i in torch.unique(ImbalancedDatasetSampler(self.dataset).weights).tolist()]
        self.assertEqual(unqi_class_weights, [0.004, 0.011])
        self.assertEqual(class_0_count/sum_count > 0.4, True)
        self.assertEqual(class_1_count/sum_count > 0.4, True)
        print(class_0_count, class_1_count)


class TestClassificationTrainer(unittest.TestCase):
    def setUp(self):
        self.dataset_path = 'cat_vs_dog_test_dataset'
        self.folders_structure = {
            "models_folder": str(Path(__file__).parent / "models"),
            "confusion_matrix_folder": str(Path(__file__).parent / "confusion_matrix"),
            "test_img_folder": str(Path(__file__).parent / "test_img_folder"),
            "metadata_json_folder": str(Path(__file__).parent / "metadata_json")
        }
        self.models_types = ['resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152', 'vgg16']
    
    def test_folder_structure_creation(self):
        model = ClassificationTrainer(dataset_path=self.dataset_path)
        for key, path in self.folders_structure.items():
            self.assertEqual(os.path.exists(path), True)
        
    def test_model_loading(self):
        for model_type in self.models_types:
            model = ClassificationTrainer(dataset_path=self.dataset_path, model_type=model_type)
            self.assertEqual(model.model_type, model_type)
            self.assertEqual(model.model.name, model_type)

    def test_get_number_of_classes(self):
        model = ClassificationTrainer(dataset_path=self.dataset_path)
        self.assertEqual(model._get_number_of_classes(), 2)

    def test_split_dataset_to_dataloaders_and_return_classes(self):
        hparams = {
            'epochs_num': 100,
            'batch_size': 32,
            'lr': 0.001,
            "train_valid_test_split": [0.8, 0.1, 0.1]
        }
        model = ClassificationTrainer(hparams=hparams, dataset_path=self.dataset_path)
        train_dataset_len = len(model.train_set)
        valid_dataset_len = len(model.valid_set)
        test_dataset_len = len(model.test_set)
        full_dataset_len = train_dataset_len + valid_dataset_len + test_dataset_len
        self.assertEqual(round(train_dataset_len/full_dataset_len, 1), hparams['train_valid_test_split'][0])
        self.assertEqual(round(valid_dataset_len/full_dataset_len, 1), hparams['train_valid_test_split'][1])
        self.assertEqual(round(test_dataset_len/full_dataset_len, 1), hparams['train_valid_test_split'][2])

    def test_forward(self):
        rand_tensor = torch.rand(1, 3, 224, 224)
        model = ClassificationTrainer(dataset_path=self.dataset_path)
        model_output = model(rand_tensor)
        self.assertEqual(model_output.shape, torch.Size([1, 2]))
        
    def test_save_metadata(self):
        model = ClassificationTrainer(dataset_path=self.dataset_path)
        model.save_metadata()
        metadata_json_path = os.path.join(self.folders_structure['metadata_json_folder'], 'metadata.json')
        self.assertEqual(os.path.exists( metadata_json_path ), True)
        with open(metadata_json_path, 'r') as JSON:
            json_dict = json.load(JSON)
        self.assertEqual(isinstance(json_dict['hparams'], dict) , True)
        self.assertEqual(isinstance(json_dict['folders_structure'], dict) , True)
        self.assertEqual(isinstance(json_dict['model_type'], str) , True)
        self.assertEqual(isinstance(json_dict['dataset_path'], str) , True)

    def test_test_epoch_end(self):
        model = ClassificationTrainer(dataset_path=self.dataset_path)
        trainer = Trainer(gpus=0, benchmark=True, 
                    max_epochs=3, default_root_dir='',
                    check_val_every_n_epoch=1,
                    )
        trainer.test(model)
        # test test img save
        image_path = os.path.join(self.folders_structure['test_img_folder'], '0_0.jpg')
        self.assertEqual(os.path.exists(image_path) , True)
        # test conf matix save
        confusion_matrix_path = os.path.join(self.folders_structure['confusion_matrix_folder'], 'conf_matrix_plot.jpg')
        self.assertEqual(os.path.exists(confusion_matrix_path) , True)
        # test save things in metadata json
        metadata_json_path = os.path.join(self.folders_structure['metadata_json_folder'], 'metadata.json')
        self.assertEqual(os.path.exists( metadata_json_path ), True)
        with open(metadata_json_path, 'r') as JSON:
            json_dict = json.load(JSON)
        self.assertEqual(isinstance(json_dict['y_true'], list) , True)
        self.assertEqual(isinstance(json_dict['y_pred'], list) , True)
        self.assertEqual(isinstance(json_dict['conf_matrix'], list) , True)
        self.assertEqual(isinstance(json_dict['test_accuracy'], float) , True)
        self.assertEqual(isinstance(json_dict['test_loss'], float) , True)

    # def tearDown(self):
    #     for key, path in self.folders_structure.items():
    #         os.remove(path)
    

if __name__ == '__main__':
    unittest.main(TestClassificationTrainer())
