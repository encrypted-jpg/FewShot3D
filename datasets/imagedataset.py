import os
import torch.utils.data as data
import json
import torch
import numpy as np
from constants import rotation_sets
from fileio import IO
import torchvision.transforms as transforms


class ImageDataset(data.Dataset):
    def __init__(self, folder, jsonfile, transform=None, mode='train'):
        self.folder = folder
        self.jsonfile = jsonfile
        self.transform = transforms.Compose(transform)
        self.file_list = []
        print(f'[DATASET] Open file {self.jsonfile}')
        with open(self.jsonfile, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            for model in value[mode]:
                _base = os.path.join(self.folder, key, model)
                for rotation in rotation_sets:
                    self.file_list.append({
                        'img1': os.path.join(_base, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_0.png"),
                        'img2': os.path.join(_base, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_1.png"),
                        'taxonomy_id': key,
                        'model_id': model,
                    })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        img1 = sample['img1']
        img2 = sample['img2']
        img1 = self.transform(IO.get(img1))
        img2 = self.transform(IO.get(img2))
        return sample['taxonomy_id'], sample['model_id'], (img1, img2)

    def __len__(self):
        return len(self.file_list)
