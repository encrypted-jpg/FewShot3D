import os
import torch.utils.data as data
import json
import torch
import numpy as np
from fileio import IO
import torchvision.transforms as transforms
from functools import lru_cache
from .constants import CACHE_SIZE


class DiffImageDataset(data.Dataset):
    def __init__(self, folder, jsonfile, transform=None, mode='train', b_tag="depth", img_height=224, img_width=224, img_count=3):
        self.folder = folder
        self.jsonfile = jsonfile
        self.transform = transforms.Compose(transform)
        self.file_list = []
        self.cache = {}
        self.cache_size = 40000
        self.img_height = img_height
        self.img_width = img_width
        print(f'[DATASET] Open file {self.jsonfile}')
        with open(self.jsonfile, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            for model in value[mode]:
                _image_base = os.path.join(self.folder, "image", key, model)
                _depth_base = os.path.join(self.folder, b_tag, key, model)
                for idx in range(img_count):
                    self.file_list.append({
                        'taxonomy_id': key,
                        'model_id': model,
                        'img1': os.path.join(_image_base, f'{idx:02d}.png'),
                        'img2': os.path.join(_depth_base, f'{idx:02d}.png')
                    })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    @lru_cache(maxsize=CACHE_SIZE)
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        sample = self.file_list[idx]
        img1 = sample['img1']
        img2 = sample['img2']
        # img1 = self.transform(IO.get(img1))
        # img2 = self.transform(IO.get(img2))
        img1 = np.array(IO.get(img1).resize(
            (self.img_height, self.img_width))).astype(np.float32) / 255.0
        img2 = np.array(IO.get(img2).resize(
            (self.img_height, self.img_width))).astype(np.float32) / 255.0
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        return sample['taxonomy_id'], sample['model_id'], (img1, img2)

    def __len__(self):
        return len(self.file_list)
