import os
import torch.utils.data as data
import json
import torch
import numpy as np
from fileio import IO
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder


class PCNDataset(data.Dataset):
    def __init__(self, folder, jsonfile, transform=None, mode='train', b_tag="depth", img_height=224, img_width=224):
        self.folder = folder
        self.jsonfile = jsonfile
        self.transform = transforms.Compose(transform)
        self.file_list = []
        self.cache = {}
        self.img_height = img_height
        self.img_width = img_width
        print(f'[DATASET] Open file {self.jsonfile}')
        with open(self.jsonfile, 'r') as f:
            data = json.load(f)

        self.setup_encoding(list(data.keys()))

        for key, value in data.items():
            for model in value[mode]:
                _file = os.path.join(
                    self.folder, "complete", key, model+".npy")
                self.file_list.append({
                    'taxonomy_id': self.le_dict[key],
                    'model_id': model,
                    'pc': _file
                })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def setup_encoding(self, keys):
        le = LabelEncoder()
        self.le_dict = dict(zip(keys, le.fit_transform(keys)))

    def __getitem__(self, idx):
        # if idx in self.cache:
        #     return self.cache[idx]
        sample = self.file_list[idx]
        pc = sample['pc']
        pc = IO.get(pc)
        pc2 = np.copy(pc)
        pc = torch.from_numpy(pc).float()
        pc2 = torch.from_numpy(pc2).float()
        # self.cache[idx] = (sample['taxonomy_id'],
        #                    sample['model_id'], (pc, pc2))
        return sample['taxonomy_id'], sample['model_id'], (pc, pc2)

    def __len__(self):
        return len(self.file_list)
