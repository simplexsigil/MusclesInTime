import os.path as osp
from typing import Callable, Optional
import pandas as pd
import torch
from torch.utils import data
from musint.utils.metadata_utils import concatenate_mint_metadata, filenames


class MintDataset(data.Dataset):
    '''Implementation of the mint dataset for pytorch dataloader'''
    def __init__(self, dataset_path: str, transform: Optional[Callable] = None):
        self.dataset_path = dataset_path
        self.metadata = concatenate_mint_metadata(dataset_path)
        self.metadata = self.metadata.dropna(subset=["path_id"])
        self.metadata = self.metadata.reset_index(drop=True)
        self.metadata = self.metadata.astype({"path_id": "int64"})
        self.metadata = self.metadata.sort_values(by="path_id")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path_id = self.metadata.iloc[idx]["path_id"]
        path = self.metadata.iloc[idx]["path"]

        # Load the muscle activations
        muscle_activations_file = osp.join(self.dataset_path, path, filenames["muscle_activations"])
        muscle_activations = pd.read_pickle(muscle_activations_file)

        # Load the ground reaction forces
        grf_file = osp.join(self.dataset_path, path, filenames["grf"])
        grf = pd.read_pickle(grf_file)

        # Load the muscle forces
        forces_file = osp.join(self.dataset_path, path, filenames["forces"])
        forces = pd.read_pickle(forces_file)

        sample = {
            "muscle_activations": muscle_activations,
            "grf": grf,
            "forces": forces,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample