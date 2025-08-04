import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

class FireSmokeDataset(Dataset):
    def __init__(self, data_path, img_size=640, augment=True, mode='train'):
        self.data_path = data_path
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        self.mode = mode
        
        # Load image and label paths
        self.img_files = []
        self.label_files = []
        
        img_dir = os.path.join(data_path, mode, 'images')
        label_dir = os.path.join(data_path, mode, 'labels')
        
        if os.path.exists(img_dir):
            for img_file in sorted(os.listdir(img_dir)):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(img_dir, img_file)
                    label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                    
                    self.img_files.append(img_path)
                    self.label_files.append(label_path)
        
        # Define augmentations
        self.transform = self._get_transforms()
        
        # Class names
        self.classes = ['fire', 'smoke', 'wall_clean', 'wall_dirty', 'wall_damaged']
        
    def _get_transforms(self):
        if self.augment:
            return A.Compose([
                A.RandomResizedCrop(height=self.img_size, width=self.img_size, scale=(0.5, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.RandomRotate90(p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.label_files[idx]
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        labels.append([x_center, y_center, width, height, int(class_id)])
        
        # Convert to numpy array
        if labels:
            labels = np.array(labels)
            bboxes = labels[:, :4]
            class_labels = labels[:, 4].astype(int)
        else:
            bboxes = np.zeros((0, 4))
            class_labels = np.zeros((0,), dtype=int)
        
        # Apply transformations
        transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        img = transformed['image']
        bboxes = transformed['bboxes']
        class_labels = transformed['class_labels']
        
        # Convert back to label format
        if len(bboxes) > 0:
            labels = np.column_stack((class_labels, bboxes))
        else:
            labels = np.zeros((0, 5))
        
        return img, torch.from_numpy(labels).float()
    
    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        
        # Stack images
        imgs = torch.stack(imgs, 0)
        
        # Add batch index to labels
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                label = torch.cat((torch.full((label.shape[0], 1), i), label), 1)
            labels[i] = label
        
        return imgs, torch.cat(labels, 0)