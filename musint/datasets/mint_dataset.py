from typing import Tuple

import pandas as pd
from torch.utils import data

from musint.utils.dataframe_utils import trim_mint_dataframe, frame_to_time
from musint.utils.metadata_utils import concatenate_mint_metadata, load_pkl_file

import os.path as osp
import os


class MintData:
    """
    A class to represent the muscle activations, grf and forces of a sample from the MINT dataset
    """

    def __init__(self, sample: pd.Series, dataset_path: str):
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
        self.gender = sample["gender"]
        self.analysed_dur = sample["analysed_dur"]
        self.analysed_percentage = sample["analysed_%"]
        self.has_gap = sample["gap"]
        self.path_id = osp.join(sample["subject"], sample["sequence"]).replace("_poses", "")

        self.muscle_activations = load_pkl_file(self.dataset_path, self.data_path, "muscle_activations")
        self.grf = load_pkl_file(self.dataset_path, self.data_path, "grf")
        self.forces = load_pkl_file(self.dataset_path, self.data_path, "forces")

        self.end_time = self.muscle_activations.index[-1]
        self.start_time = self.muscle_activations.index[0]
        self.num_frames = self.muscle_activations.shape[0]
        self.fps = 50.0  # all the samples in the dataset have a fixed fps of 50.0

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

    def get_valid_indices(self, frame_window: Tuple[int, int], fps=20.0):
        """
        Get the valid indices of the muscle activations given the frame indices and their corresponding fps

        Returns:
        np.ndarray: The valid indices
        """
        time_window = (frame_to_time(frame_window[0], fps), frame_to_time(frame_window[1], fps))
        trimmed_muscle_activations = self.get_muscle_activations(time_window, fps, as_numpy=False)

        # Get the indices that are within the frame_window range
        valid_indices = trimmed_muscle_activations.index

        return valid_indices

    def get_gaps(self, as_frame=False, fps=50.0):
        """
        Gets all the pairs of indices before and after a gap in the muscle activations
        """
        differences = self.get_muscle_activations(None, fps, as_numpy=False).index.to_series().diff()

        normal_difference = round(1 / fps, 2) + 0.01
        # Get the indices of the differences that are larger than 0.02s
        gap_indices = differences[differences > normal_difference].index

        # Create a list of tuples, where each tuple contains the index before and after the gap
        gap_tuples = []
        for gap_index in gap_indices:
            pos = self.muscle_activations.index.get_loc(gap_index)
            if pos > 0:  # Skip if it's the first index
                if as_frame:
                    gap_tuples.append((pos - 1, pos))
                else:
                    gap_tuples.append((self.muscle_activations.index[pos - 1], self.muscle_activations.index[pos]))

        return gap_tuples


class MintDataset(data.Dataset):
    """
    A PyTorch dataset for the MINT dataset

    Parameters:
    dataset_path (str): The path to the dataset
    transform (Optional[Callable]): A transform to apply to the data
    """

    def __init__(self, dataset_path: str, use_cache: bool = True):
        self.dataset_path = dataset_path
        self.metadata = concatenate_mint_metadata(dataset_path, delete_old=not use_cache)

    def __len__(self):
        # return the number of samples of the dataframe (metadata)
        return self.metadata.shape[0]

    def __getitem__(self, idx: int) -> MintData:
        """Get a sample by its index"""
        sample = self.metadata.iloc[idx]
        return MintData(sample, self.dataset_path)

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


if __name__ == "__main__":
    """Example usage of the MintDataset class"""
    dataset_path = "/lsdf/data/activity/MuscleSim/musclesim_dataset"
    mint_dataset = MintDataset(dataset_path, use_cache=False)
    sample = mint_dataset[0]
    print(sample.get_forces(None))
    sample_by_path_id = mint_dataset.by_path_id("s1/acting2")
    print(sample_by_path_id.fps)
    sample_by_path = mint_dataset.by_path("TotalCapture/TotalCapture/s1/acting2_poses")
    print(sample_by_path.path_id)
    sample_by_babel_sid = mint_dataset.by_babel_sid(12906)
    print(sample_by_babel_sid.get_forces(None))
    sample_by_subject_and_sequence = mint_dataset.by_subject_and_sequence("s1", "acting2_poses")
    print(sample_by_subject_and_sequence.get_forces(None))
    sample_by_humanml3d_name = mint_dataset.by_humanml3d_name("003260")
    print(sample_by_humanml3d_name[0].get_forces(None))
    sample_by_humanml3d_name = mint_dataset.by_humanml3d_name("003260.npy")
    print(sample_by_humanml3d_name[0].get_forces(None))
    sample_by_humanml3d_name = mint_dataset.by_humanml3d_name("003260.npy")
    print(sample_by_humanml3d_name[0].get_forces(None))
    print("Index values:")
    pd.set_option("display.max_rows", None)

    print(sample_by_humanml3d_name[0].get_forces(None, target_fps=50).index.values)
    print(sample_by_humanml3d_name[0].get_valid_indices((10, 1000), 20.0))
    print(sample_by_humanml3d_name[0].get_gaps())
    print(sample_by_humanml3d_name[0].has_gap)

    print(sample_by_humanml3d_name[0].muscle_activations.index)
