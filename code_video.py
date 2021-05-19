"""
source:
https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning
https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
https://medium.com/pytorch/introduction-to-captum-a-model-interpretability-library-for-pytorch-d236592d8afa
https://en.wikipedia.org/wiki/Multinomial_distribution
"""

import torch
import cv2
import os
import json
import torch.nn as nn
import numpy as np
from pathlib import Path
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import itertools


class AlbumentationsTransform:
    def __init__(self):
        self.img_transforms = A.Compose(
                            [
                                A.Resize(224, 224), A.RGBShift(), A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.2), A.ChannelShuffle(0.2), A.ColorJitter(p=0.5),
                                A.Cutout(num_holes=3, max_h_size=24, max_w_size=24, 
                                        fill_value=0, always_apply=False, p=0.5),
                                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                                A.PadIfNeeded(min_height=224, min_width=224, always_apply=True, border_mode=0),
                                A.IAAAdditiveGaussianNoise(p=0.2),
                                A.IAAPerspective(p=0.5),
                                A.RandomBrightnessContrast(p=0.5),
                                A.OneOf(
                                    [
                                        A.CLAHE(p=1),
                                        A.RandomBrightness(p=1),
                                        A.RandomGamma(p=1),
                                    ],
                                    p=0.9,
                                ),
                                A.OneOf(
                                    [
                                        A.IAASharpen(p=1),
                                        A.Blur(blur_limit=3, p=1),
                                        A.MotionBlur(blur_limit=3, p=1),
                                    ],
                                    p=0.9,
                                ),
                                A.OneOf(
                                    [
                                        A.RandomContrast(p=1),
                                        A.HueSaturationValue(p=1),
                                    ],
                                    p=0.9,
                                ),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensorV2(),
                            ])
    def __call__(self, img):
        img = np.array(img)
        return self.img_transforms(image = img).copy()


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx][1]

    def __iter__(self):
        # torch.multinomial - Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution
        # located in the corresponding row of tensor input.
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class ClassificationTrainer(pl.LightningModule):
    def __init__(self, hparams = {
            'epochs_num': 100,
            'batch_size': 32,
            'lr': 0.001,
            "train_valid_test_split": [0.8, 0.1, 0.1],
        }, model_type='resnet50',
        dataset_path = 'dataset',
        folders_structure = {
            "models_folder": str(Path(__file__).parent / "models"),
            "confusion_matrix_folder": str(Path(__file__).parent / "confusion_matrix"),
            "test_img_folder": str(Path(__file__).parent / "test_img_folder"),
            "metadata_json_folder": str(Path(__file__).parent / "metadata_json")
        }):
            super().__init__()
            self._hparams = hparams
            self.model_type = model_type
            self.dataset_path = dataset_path
            self.folders_structure = folders_structure
            self.model_metadata = {
                'hparams': hparams,
                'model_type':model_type,
                'dataset_path': dataset_path, 
                'folders_structure': folders_structure
            }

            self.img_transform = transforms.Compose([AlbumentationsTransform()])
            self.criterion = nn.CrossEntropyLoss()

            self.model = self._load_specific_model()
            self._split_dataset_to_dataloaders_and_return_classes()
            self._create_dir_structure()

            self.last_best_valid_error = 10000000000.0
            self.test_images = []
            self.test_losses = []
            self.test_true = []
            self.test_pred = []

    def _load_specific_model(self):
        number_of_classes = self._get_number_of_classes()
        if self.model_type == 'resnet50':
            model = models.resnet50(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, number_of_classes)).to(self.device)
            model = model.to(self.device)
            model.name = 'resnet50'
        elif self.model_type == 'resnet18':
            model = models.resnet18(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, number_of_classes)).to(self.device)
            model = model.to(self.device)
            model.name = 'resnet18'
        elif self.model_type == 'resnet34':
            model = models.resnet34(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, number_of_classes)).to(self.device)
            model = model.to(self.device)
            model.name = 'resnet34'
        elif self.model_type == 'resnet101':
            model = models.resnet101(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, number_of_classes)).to(self.device)
            model = model.to(self.device)
            model.name = 'resnet101'
        elif self.model_type == 'resnet152':
            model = models.resnet152(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, number_of_classes)).to(self.device)
            model = model.to(self.device)
            model.name = 'resnet152'
        elif self.model_type == 'vgg16':
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = nn.Sequential(
                        nn.Linear(25088, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, number_of_classes)).to(self.device)
            model = model.to(self.device)
            model.name = 'vgg16'
        return model

    def _get_number_of_classes(self):
        return len([ f.path for f in os.scandir(self.dataset_path) if f.is_dir() ])

    def _split_dataset_to_dataloaders_and_return_classes(self):
        dataset = datasets.ImageFolder(self.dataset_path, self.img_transform)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.model_metadata.update(
            {
                'classes': self.classes,
                'class_to_idx': self.class_to_idx,
            }
        )
        train_size = int(self._hparams['train_valid_test_split'][0]*len(dataset))
        valid_size = int(self._hparams['train_valid_test_split'][1]*len(dataset))
        test_size = int(self._hparams['train_valid_test_split'][2]*len(dataset))
        rest = len(dataset) - train_size - valid_size - test_size
        train_size += rest
        train_set, valid_set, test_set = torch.utils.data.random_split(dataset,
                                            [train_size, valid_size, test_size])
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

    def _create_dir_structure(self):
        for _, path in self.folders_structure.items():
            os.makedirs(path, exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(self.train_set,
                                                        sampler=ImbalancedDatasetSampler(self.train_set),
                                                        batch_size=self._hparams['batch_size'])
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(self.valid_set, sampler=ImbalancedDatasetSampler(self.valid_set),
                                                batch_size=self.hparams['batch_size'])
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(self.test_set, sampler=ImbalancedDatasetSampler(self.test_set),
                                                batch_size=self.hparams['batch_size'])
        return test_dataloader

    def training_step(self, batch, batch_nb):
        x,y = batch
        if isinstance(x, dict):
            x = x['image']
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x,y = batch
        if isinstance(x, dict):
            x = x['image']
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('valid_loss', loss)

        ps = torch.exp(pred)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        self.log('accuracy_loss', accuracy)
        if loss < self.last_best_valid_error:
            self.last_best_valid_error = loss
            self.save_model(loss=loss.item(), acc=accuracy, mode='valid')
        return loss

    def test_step(self, batch, batch_nb):
        x,y = batch
        if isinstance(x, dict):
            x = x['image']
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('test_loss_per_batch', loss)

        ps = torch.exp(pred)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        self.log('test_accuracy_per_batch', accuracy)

        self.test_images.append(x)
        self.test_losses.append(loss.item())
        self.test_true += y.tolist()
        self.test_pred += top_class.tolist()
        return loss

    def test_epoch_end(self, outputs):
        test_img_paths = self.save_and_convert_test_images_to_paths(self.test_images)
        self.test_pred = [ i[0] for i in self.test_pred ]

        loss = sum(self.test_losses) / len(self.test_losses)
        self.log('test_loss_per_epoch', loss)

        accuracy = accuracy_score(self.test_true, self.test_pred)
        self.log('test_accuracy_per_epoch', accuracy)

        conf_matrix = confusion_matrix(self.test_true, self.test_pred)
        conf_matrix_path = os.path.join(self.folders_structure['confusion_matrix_folder'], 'conf_matrix_plot.jpg')
        plot_confusion_matrix(conf_matrix, conf_matrix_path, self.classes)

        self.model_metadata.update(
            {
                'y_true': self.test_true,
                'y_pred': self.test_pred,
                'test_accuracy': accuracy,
                'test_loss': loss,
                'conf_matrix': conf_matrix.tolist()
            }
        )
        self.save_metadata()
        self.save_model(loss=loss, acc=accuracy, mode='test')
        return 

    def save_and_convert_test_images_to_paths(self, test_images):
        test_images_paths = []
        for batch_idx , test_image in enumerate(test_images):
            if test_image.shape[0] > 1:
                for image_idx , i_image in enumerate(test_image):
                    i_image = self._inv_normalize_tensor(i_image)
                    img_path = os.path.join(self.folders_structure['test_img_folder'], str(batch_idx) + '_'+ str(image_idx)+'.jpg')
                    torchvision.utils.save_image(i_image, img_path)
                    test_images_paths.append(img_path)
            else:
                test_image = self._inv_normalize_tensor(test_image)
                img_path = os.path.join(self.folders_structure['test_img_folder'], str(batch_idx) + '.jpg')
                torchvision.utils.save_image(test_image, img_path)
                test_images_paths.append(img_path)
        return test_images_paths

    def _inv_normalize_tensor(self, tensor):
        tensor = torch.squeeze(tensor, 0)
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        return inv_normalize(tensor)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self._hparams['lr'])

    def save_metadata(self):
        json_path = os.path.join(self.folders_structure['metadata_json_folder'], 'metadata.json')
        with open(json_path, 'w') as file:
            json.dump(self.model_metadata, file)

    def save_model(self, loss=0.0, acc=0.0, mode='valid'):
        model_name = mode+'_'+'loss_' + str(round(loss, 4)) + '_accuracy_' + str(round(acc, 4)) + '.pth'
        model_save_path = os.path.join(self.folders_structure['models_folder'], model_name)
        torch.save(self.model, model_save_path)

def plot_confusion_matrix(cm, save_path,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    fig.savefig(save_path, dpi=fig.dpi)
    return fig


if __name__ == '__main__':
    torch.cuda.empty_cache()
    dataset_path = 'cat_vs_dog_test_dataset'
    model1 = ClassificationTrainer(dataset_path=dataset_path)
    checkpoint_save_path = str(Path(__file__).parent / '')
    trainer = Trainer(gpus=0, benchmark=True, 
                    max_epochs=1, default_root_dir=checkpoint_save_path,
                    check_val_every_n_epoch=1,
                    # resume_from_checkpoint=checkpoint_path 
                    )
    trainer.test(model1)