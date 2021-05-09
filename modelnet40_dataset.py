import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from transformations import CoordinateTransformation

class ModelNet40(torch.utils.data.Dataset):
    def __init__(self, DIR_PATH, phase = "train", transform = None):
        torch.utils.data.Dataset.__init__(self)

        self.phase = phase
        self.transform = transform

        train_data = np.load(os.path.join(DIR_PATH, 'train.npy'), allow_pickle=True)
        test_data = np.load(os.path.join(DIR_PATH, 'test.npy'), allow_pickle=True)
        self.train_points = train_data.item()['points']
        self.train_labels = train_data.item()['labels']
        self.test_points = test_data.item()['points']
        self.test_labels = test_data.item()['labels']
        self.map = test_data.item()['map']
            
    def __len__(self):
        if self.phase == "train":
            return len(self.train_points)
        else:
            return len(self.test_points)

    def __getitem__(self, idx):
        if self.phase=="train":
            points = self.train_points[idx]
            label = self.train_labels[idx]
        else:
            points = self.test_points[idx]
            label = self.test_labels[idx]
        scaler = MinMaxScaler((-1, 1))
        points = scaler.fit_transform(points)
        if self.transform is not None:
        	points = self.transform.apply_transformation(points)
        coordinates = points.astype('float32')
        np.random.shuffle(points)
        return {
            "coordinates": torch.from_numpy(coordinates),
            "features": torch.from_numpy(coordinates),
            "labels": label
        }