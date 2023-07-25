import os
import cv2
import torchstain
import h5py
import torch
from torch.utils.data import Dataset

class PCAMDataset(Dataset):
    def __init__(self, file_path, label_path = None, transform=None, normalize=None):
        self.dataset = h5py.File(file_path, 'r')
        self.images = self.dataset['x']
        self.transform = transform
        self.normalize = normalize
        self.labels =  h5py.File(label_path, 'r')['y'] if label_path else None

        if self.transform and self.normalize:
            if self.normalize == 'macenko':
                self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
            elif self.normalize == 'reinhard':
                self.normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
            target_image = cv2.cvtColor(cv2.imread(os.path.join('..', 'data', 'norm_better.png')), cv2.COLOR_BGR2RGB)
            self.normalizer.fit(self.transform(target_image))
    
    def get_normalization(self, image):
        if self.normalize == 'macenko':
            # print('Beefore:', image[0,])
            image, _, _ = self.normalizer.normalize(I = image)
            # image = image.type(torch.float)  # type: ignore
            # print('After:', image[0,])
        elif self.normalize == 'reinhard':
            #This METHOD GIVES torch.uint8 instead of float!
            image = self.normalizer.normalize(I= image)
            image = image.type(torch.float)/255.0 # type: ignore
        return image

    def __len__(self):
        return len(self.images) # type: ignore
    
    def __getitem__(self, idx):
        data = self.images[idx] # type: ignore
        # Apply transformation if provided
        if self.transform:
            data = self.transform(data)
        if self.normalize:
            data = self.get_normalization(data).permute(2, 0, 1) # type: ignore
        if self.labels:
            labels = self.labels[idx] # type: ignore
            return data, labels
        return data