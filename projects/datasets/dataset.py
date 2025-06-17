import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, root, transform, read_file,
                 in_memory=False, take: int = -1):
        super(Dataset, self).__init__()
        self.root = root
        self.filelist = glob.glob(f"{root}/*_partial.ply")
        self.transform = transform
        self.in_memory = in_memory
        self.read_file = read_file
        self.take = take

        if self.in_memory:
            self.partial_samples = [
                self.read_file(f) for f in tqdm(self.filelist, ncols=80, leave=False)
            ]
            self.complete_samples = [
                self.read_file(f.replace("_partial.ply", "_complete.ply")) for f in tqdm(self.filelist, ncols=80, leave=False)
            ]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        partial_sample = self.partial_samples[idx] if self.in_memory else self.read_file(self.filelist[idx])
        complete_sample = self.complete_samples[idx] if self.in_memory else self.read_file(self.filelist[idx].replace("_partial.ply", "_complete.ply"))
        output = self.transform(partial_sample, complete_sample, idx)        # data augmentation + build octree
        output['label'] = 0
        output['filename'] = self.filelist[idx]
        return output

