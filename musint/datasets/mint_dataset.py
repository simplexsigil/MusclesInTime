import argparse
import ast
import os
import os.path as osp
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import random_split

from musint.utils.dataframe_utils import (
    frame_to_time,
    time_to_frame,
    trim_mint_dataframe,
)
from musint.utils.hml3d_utils.humanml3d_utils import segment_motions
from musint.utils.metadata_utils import (
    concatenate_mint_metadata,
    load_pkl_file,
)


class MintData:
    """
    A class to represent the muscle activations, grf and forces of a sample from the MINT dataset
    """

    def __init__(self, sample: pd.Series, dataset_path: str, pyarrow: bool = False, load_humanml3d_names: bool = True):
        self.dataset_path = dataset_path
        self.data_path = sample["data_path"]
        self.full_data_path = f"{dataset_path}/{self.data_path}"
        self.weight = sample["weight_kg"]
        self.height = sample["height_cm"]
        self.amass_dur = sample["amass_dur"]
        self.babel_sid = sample["babel_sid"]
        self.subject = sample["subject"]
        self.sequence = sample["sequence"]
        self.dataset = sample["dataset"]
        self.subdataset_path = osp.join(self.data_path.split("/")[0], self.data_path.split("/")[1])
        self.gender = sample["gender"]
        self.analysed_dur = sample["analysed_dur"]
        self.analysed_percentage = sample["analysed_%"]
        self.has_gap = ast.literal_eval(sample["gap"])
        self.path_id = MintData.path_id_from_sample(sample)

        self.muscle_activations = load_pkl_file(self.dataset_path, self.data_path, "muscle_activations")
        self.grf = load_pkl_file(self.dataset_path, self.data_path, "grf")
        self.forces = load_pkl_file(self.dataset_path, self.data_path, "forces")

        if pyarrow:
            self.muscle_activations = self.muscle_activations.convert_dtypes(dtype_backend="pyarrow")
            self.grf = self.grf.convert_dtypes(dtype_backend="pyarrow")
            self.forces = self.forces.convert_dtypes(dtype_backend="pyarrow")

        self.start_time = self.muscle_activations.index[0]
        self.end_time = self.muscle_activations.index[-1]
        self.num_frames = self.muscle_activations.shape[0]
        self.fps = 50.0  # all the samples in the dataset have a fixed fps of 50.0
        if load_humanml3d_names:
            self.humanml3d_source_path = self.get_humanml3d_source_path()
            self.humanml3d_name = self.get_humanml3d_names()

    @classmethod
    def path_id_from_sample(cls, sample: pd.Series):
        return osp.join(sample["subject"], sample["sequence"]).replace("_poses", "")

    def get_muscle_activations(
        self,
        time_window: Tuple[float, float] = None,
        target_fps=20.0,
        rolling_average=False,
        target_frame_count=None,
        as_numpy=False,
    ):
        """
        Resample muscle activations to the given time window and fps. Returns the values as a numpy array or a dataframe

        Parameters:
        time_window (Tuple[float, float]): The start and end times for the window
        target_fps (float): The target frames per second for resampling
        rolling_average (bool): Whether to apply a rolling average

        Returns:
        np.ndarray/pd.DataFrame: The resampled values as a numpy array or a dataframe
        """
        if time_window is None:
            time_window = (self.start_time, self.end_time)

        return trim_mint_dataframe(
            self.muscle_activations,
            time_window,
            target_fps,
            rolling_average,
            target_frame_count,
            as_numpy,
        )

    def get_grf(
        self,
        time_window: Tuple[float, float] = None,
        target_fps=20.0,
        rolling_average=False,
        target_frame_count=None,
        as_numpy=False,
    ):
        """
        Resample grf to the given time window and fps. Returns the values as a numpy array or a dataframe

        Parameters:
        time_window (Tuple[float, float]): The start and end times for the window
        target_fps (float): The target frames per second for resampling
        rolling_average (bool): Whether to apply a rolling average

        Returns:
        np.ndarray/pd.DataFrame: The resampled values as a numpy array or a dataframe
        """
        if time_window is None:
            time_window = (self.start_time, self.end_time)

        return trim_mint_dataframe(
            self.grf,
            time_window,
            target_fps,
            rolling_average,
            target_frame_count,
            as_numpy,
        )

    def get_forces(
        self,
        time_window: Tuple[float, float] = None,
        target_fps=20.0,
        rolling_average=False,
        target_frame_count=None,
        as_numpy=False,
    ):
        """
        Resample forces to the given time window and fps. Returns the values as a numpy array or a dataframe

        Parameters:
        time_window (Tuple[float, float]): The start and end times for the window
        target_fps (float): The target frames per second for resampling
        rolling_average (bool): Whether to apply a rolling average

        Returns:
        np.ndarray/pd.DataFrame: The resampled values as a numpy array or a dataframe
        """
        if time_window is None:
            time_window = (self.start_time, self.end_time)

        return trim_mint_dataframe(
            self.forces,
            time_window,
            target_fps,
            rolling_average,
            target_frame_count,
            as_numpy,
        )

    def get_valid_indices(self, time_window: Tuple[float, float] = None, target_fps=20.0, as_time=True):
        """
        Gets the valid indices of the muscle activations given the frame indices.
        Returns the indices as frames or time corresponding to the target fps.

        Returns:
        np.ndarray: The valid indices as frames or time
        """
        if time_window is None:
            time_window = (self.start_time, self.end_time)

        trimmed_muscle_activation_index = self.muscle_activations.index[
            self.muscle_activations.index.to_series().between(time_window[0], time_window[1])
        ]

        frame_indices = np.round(trimmed_muscle_activation_index * target_fps, 0).astype(int).unique()

        if as_time:
            frame_times = frame_indices / target_fps
            return np.round(frame_times, 2)
        else:
            return frame_indices

    def get_gaps(self, as_frame=False, target_fps=20.0):
        """
        Gets all the pairs of indices before and after a gap in the muscle activations
        """
        # resample index of the muscle activations to fps
        muscle_activation_index = self.muscle_activations.index
        muscle_activation_index = np.round(muscle_activation_index * self.fps).astype(int)

        differences = pd.Series(muscle_activation_index).diff()

        normal_difference = 1.01
        # Get the indices of the differences that are larger than 0.02s
        gap_indices = muscle_activation_index[differences > normal_difference]

        gap_tuples = []
        for gap_index in gap_indices:
            pos2 = gap_index
            time = frame_to_time(gap_index, self.fps)
            previous_frame = self.muscle_activations.index.get_loc(frame_to_time(pos2, self.fps)) - 1
            previous_time = self.muscle_activations.index[previous_frame]
            if as_frame:
                gap_tuples.append(
                    (
                        time_to_frame(previous_time, target_fps),
                        time_to_frame(time, target_fps),
                    )
                )
            else:
                gap_tuples.append((previous_time, time))

        return gap_tuples

    def get_humanml3d_source_path(self, data_root: str = "."):
        """
        Get the source path of the HumanML3D sample
        """
        dataset = self.data_path.split("/")[1]
        source_path = osp.join(data_root, "pose_data", dataset, self.subject, self.sequence + ".npy")

        return source_path

    def get_humanml3d_names(self):
        """
        Get the humanml3d names of the sample
        """
        csv_path = osp.join(os.path.dirname(os.path.realpath(__file__)), "humanml3d_index.csv")
        df = pd.read_csv(csv_path)

        humanml3d_names = df[df["source_path"] == self.humanml3d_source_path]["new_name"].values

        return humanml3d_names

    def __repr__(self):
        attributes = [
            "path_id",
            "babel_sid",
            "dataset",
            "amass_dur",
            "num_frames",
            "fps",
            "analysed_dur",
            "analysed_percentage",
            "dataset_path",
            "data_path",
            "full_data_path",
            "weight",
            "height",
            "subject",
            "sequence",
            "gender",
            "has_gap",
            "humanml3d_source_path",
            "humanml3d_name",
        ]
        info = [f"{attr}={getattr(self, attr)!r}" for attr in attributes]
        return f"{self.__class__.__name__}({', '.join(info)})"


class MintDataset(data.Dataset):
    """
    A PyTorch dataset for the MINT dataset

    Parameters:
    dataset_path (str): The path to the dataset
    transform (Optional[Callable]): A transform to apply to the data
    """

    def __init__(
        self,
        dataset_path: str,
        use_cache: bool = True,
        keep_in_memory: bool = False,
        pyarrow: bool = False,
        load_humanml3d_names: bool = True,
    ):
        self.dataset_path = dataset_path
        self.metadata = concatenate_mint_metadata(dataset_path, delete_old=not use_cache)
        self.keep_in_memory = keep_in_memory
        self.memorized_samples = {}
        self.pyarrow = pyarrow
        self.load_humanml3d_names = load_humanml3d_names
        self.__path_indexer = PathIdIndexer(self)

    def __len__(self):
        # return the number of samples of the dataframe (metadata)
        return self.metadata.shape[0]

    def __getitem__(self, idx: int) -> MintData:
        """Get a sample by its index"""
        path_id = self.path_ids[idx]
        if self.keep_in_memory and path_id in self.memorized_samples:
            return self.memorized_samples[path_id]

        sample = self.metadata.iloc[idx]
        mint_data = MintData(
            sample, self.dataset_path, pyarrow=self.pyarrow, load_humanml3d_names=self.load_humanml3d_names
        )

        if self.keep_in_memory:
            self.memorized_samples[path_id] = mint_data

        return mint_data

    @property
    def path_ids(self):
        return self.__path_indexer

    def by_path_id(self, path_id: str):
        """
        Get a sample by its path_id

        Parameters:
        path_id (int): The path_id of the sample

        Returns:
        MintData: The sample
        """
        if path_id not in self.metadata.index:
            raise ValueError(f"No sample found with path_id: {path_id}")
        return self[self.metadata.index.get_loc(path_id)]

    def by_path(self, path: str):
        """
        Get a sample by its path

        Parameters:
        path (str): The path of the sample

        Returns:
        MintData: The sample
        """
        filtered = self.metadata[self.metadata["data_path"] == path]
        if filtered.empty:
            raise ValueError(f"No sample found with path: {path}")
        idx = self.metadata.index.get_loc(filtered.index[0])
        return self[idx]

    def by_babel_sid(self, babel_sid: int):
        """
        Get a sample by its babel_sid

        Parameters:
        babel_sid (int): The babel_sid of the sample

        Returns:
        MintData: The sample
        """
        filtered = self.metadata[self.metadata["babel_sid"] == babel_sid]
        if filtered.empty:
            raise ValueError(f"No sample found with babel_sid: {babel_sid}")
        idx = self.metadata.index.get_loc(filtered.index[0])
        return self[idx]

    def by_subject_and_sequence(self, subject: str, sequence: str):
        """
        Get a sample by its subject and sequence

        Parameters:
        subject (str): The subject of the sample
        sequence (str): The sequence of the sample

        Returns:
        MintData: The sample
        """
        filtered = self.metadata[self.metadata["subject"] == subject]
        filtered = filtered[filtered["sequence"] == sequence]
        if filtered.empty:
            raise ValueError(f"No sample found with subject: {subject} and sequence: {sequence}")
        idx = self.metadata.index.get_loc(filtered.index[0])
        return self[idx]

    def by_humanml3d_name(self, humanml3d_name: str, as_time=False):
        """
        Get a sample by its humanml3d_name

        Parameters:
        humanml3d_name (str): The humanml3d_name of the sample

        Returns:
        MintData: The sample
        (Tuple[float, float]): The start and end times of the sample if as_time is True else the start and end frames of the HumanML3D sample
        """
        csv_path = osp.join(os.path.dirname(os.path.realpath(__file__)), "humanml3d_index.csv")
        df = pd.read_csv(csv_path)

        if not humanml3d_name.endswith(".npy"):
            humanml3d_name = humanml3d_name + ".npy"

        if humanml3d_name.startswith("M"):
            humanml3d_name = humanml3d_name[1:]

        if humanml3d_name not in df["new_name"].values:
            raise ValueError(f"No sample found with humanml3d_name: {humanml3d_name}")

        row = df[df["new_name"] == humanml3d_name]
        source_path = row["source_path"].values[0]

        subject = source_path.split("/")[3]
        sequence = source_path.split("/")[4].replace(".npy", "")

        frames = (row["start_frame"].values[0], row["end_frame"].values[0])

        frame_times = (
            frame_to_time(frames[0], 20.0),
            frame_to_time(frames[1], 20.0),
        )  # 20.0 is the fps for the humanml3d dataset

        if as_time:
            return self.by_subject_and_sequence(subject, sequence), frame_times
        else:
            return self.by_subject_and_sequence(subject, sequence), frames

    def generate_segment_data(self, data_root: str, save_dir: str, csv_file: str, dataset=["val"]):
        """
        Generates segments of motion data and saves them as npy files in the save_dir from the pose_data of AMASS dataset.
        The motion data segments are generated based on the availability of the mint data.

        Args:
            mint_dataset (MintDataset): The MintDataset object containing the dataset information.
            save_dir (str): Directory path where the generated segments will be saved.
            csv_file (str): File path of the CSV file where segment information will be written.

        Returns:
            None
        """
        generator = torch.Generator()
        generator.manual_seed(0)
        train_size = int(0.7 * len(self))
        val_size = int(0.2 * len(self))
        test_size = len(self) - train_size - val_size

        train, val, test = random_split(dataset=self, lengths=[train_size, val_size, test_size], generator=generator)

        for dataset in dataset:
            if dataset == "train":
                data = train
            elif dataset == "val":
                data = val
            elif dataset == "test":
                data = test

            segment_motions(data_root, save_dir, csv_file, data, dataset)

    def generate_chat_prompts(self, save_dir: str, path: str, start_frame: int, end_frame: int, code):
        """
        Generates chat prompts for the MINT dataset

        Args:
            save_dir (str): Directory path where the generated chat prompts will be saved.

        Returns:
            None
        """
        dir_name, file_name = os.path.split(path)
        file_name = file_name.removesuffix(".npy")
        if file_name.startswith("M_"):
            file_name = file_name[2:]
        sample_path = os.path.join(dir_name, file_name[:-2])

        mint_data = self.by_path(sample_path)
        time_window = (frame_to_time(start_frame, 20.0), frame_to_time(end_frame, 20.0))
        muscle_activation = mint_data.get_muscle_activations(time_window)
        grf = mint_data.get_grf(time_window)
        forces = mint_data.get_forces(time_window)

        # TODO: In progress

        print(muscle_activation)


class PathIdIndexer:
    """
    This indexer allows for iterating over path_ids without instantiating MintData elements (which includes loading the data into memory).
    """

    def __init__(self, mint_dataset: MintDataset):
        self.mint_dataset = mint_dataset

    def __getitem__(self, idx: str):
        sample = self.mint_dataset.metadata.iloc[idx]
        return MintData.path_id_from_sample(sample)


if __name__ == "__main__":
    """Example usage of the MintDataset class"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)

    dataset_path = parser.parse_args().dataset_path
    mint_dataset = MintDataset(dataset_path, use_cache=False)
    sample_by_path = mint_dataset.by_path("TotalCapture/TotalCapture/s1/acting2_poses")
    print(sample_by_path)
    path_id = mint_dataset.path_ids[0]
    print(path_id)
    sample_by_path_id = mint_dataset.by_path_id(path_id)
    sample_by_babel_sid = mint_dataset.by_babel_sid(12906)
    sample_by_humanml3d_name = mint_dataset.by_humanml3d_name("003260")[0]
    valid_indices = sample_by_humanml3d_name.get_valid_indices((10, 1000), 20.0)
