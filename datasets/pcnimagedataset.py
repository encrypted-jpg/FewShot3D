import os
import torch.utils.data as data
import json
import torch
import numpy as np
from fileio import IO
import torchvision.transforms as transforms
from functools import lru_cache
from .constants import CACHE_SIZE


class PCNImageDataset(data.Dataset):
    def __init__(self, folder, jsonfile, transform=None, mode='train', b_tag="depth", img_height=224, img_width=224, img_count=12):
        self.folder = folder
        self.jsonfile = jsonfile
        self.transform = transforms.Compose(transform)
        self.file_list = []
        self.img_height = img_height
        self.img_width = img_width
        print(f'[DATASET] Open file {self.jsonfile}')
        with open(self.jsonfile, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            for model in value[mode]:
                _file = os.path.join(
                    self.folder, "complete", key, model+".npy")
                _image_base = os.path.join(self.folder, "image", key, model)
                for idx in range(img_count):
                    self.file_list.append({
                        'taxonomy_id': key,
                        'model_id': model,
                        'img': os.path.join(_image_base, f'{idx:02d}.png'),
                        'pc': _file
                    })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    @lru_cache(maxsize=CACHE_SIZE)
    def get_pc(self, idx):
        sample = self.file_list[idx]
        pc = sample['pc']
        pc = IO.get(pc)
        pc = torch.from_numpy(pc)
        return pc

    @lru_cache(maxsize=CACHE_SIZE)
    def get_img(self, idx):
        sample = self.file_list[idx]
        img = sample['img']
        img = np.array(IO.get(img).resize(
            (self.img_height, self.img_width))).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc = self.get_pc(idx)
        img = self.get_img(idx)
        pc2 = np.copy(pc)
        pc2 = torch.from_numpy(pc2)
        return sample['taxonomy_id'], sample['model_id'], (img, pc, pc2)

    def __len__(self):
        return len(self.file_list)
