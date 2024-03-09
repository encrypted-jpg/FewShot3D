import os
import h5py
import torch
import torch.utils.data as data
import json
from functools import lru_cache
from .constants import CACHE_SIZE
from sklearn.preprocessing import LabelEncoder


class PCNImageHDF5Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        jsonfile,
        hdf5_file="dataset.h5",
        transform=None,
        mode="train",
        b_tag="depth",
        img_height=224,
        img_width=224,
        img_count=3,
    ):
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        self.img_count = img_count
        self.data = h5py.File(hdf5_file, "r")

        self.file_list = []
        with open(jsonfile, "r") as f:
            data = json.load(f)

        self.setup_encoding(list(data.keys()))

        for key in data.keys():
            for model in data[key][mode]:
                for i in range(img_count):
                    self.file_list.append(
                        {"taxonomy_id": key, "model_id": model, "img_num": i}
                    )
        print(f"[DATASET] {len(self.file_list)} instances were loaded")

    def setup_encoding(self, keys):
        le = LabelEncoder()
        self.le_dict = dict(zip(keys, le.fit_transform(keys)))

    # @lru_cache(maxsize=CACHE_SIZE)
    def get_img(self, idx, img_num):
        group_key = (
            f'{self.file_list[idx]["taxonomy_id"]}-{self.file_list[idx]["model_id"]}'
        )
        img_array = self.data[group_key][f"img_{img_num}"][()]
        img_array = torch.from_numpy(img_array)
        return img_array

    # @lru_cache(maxsize=CACHE_SIZE)
    def get_pc(self, idx):
        group_key = (
            f'{self.file_list[idx]["taxonomy_id"]}-{self.file_list[idx]["model_id"]}'
        )
        pc_array = self.data[group_key]["pc"][()]
        pc = torch.from_numpy(pc_array)
        return pc

    def __getitem__(self, idx):
        taxonomy_id = self.file_list[idx]["taxonomy_id"]
        model_id = self.file_list[idx]["model_id"]
        img_idx = self.file_list[idx]["img_num"]
        img = self.get_img(idx, img_idx)
        pc = self.get_pc(idx)

        # pc2 = np.copy(pc)
        # pc2 = torch.from_numpy(pc2)
        return self.le_dict[taxonomy_id], model_id, (img, pc, pc)

    def __len__(self):
        return len(self.file_list)
