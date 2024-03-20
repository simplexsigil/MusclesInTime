from typing import Tuple

import pandas as pd
from torch.utils import data

from musint.utils.dataframe_utils import trim_mint_dataframe
from musint.utils.metadata_utils import concatenate_mint_metadata, load_pkl_file


class MintData:
    """
    A class to represent the muscle activations, grf and forces of a sample from the MINT dataset
    """

    def __init__(
        self, muscle_activations: pd.DataFrame, grf: pd.DataFrame, forces: pd.DataFrame
    ):
        self.muscle_activations = muscle_activations
        self.grf = grf
        self.forces = forces
        self.end_time = self.muscle_activations.index[-1]
        self.start_time = self.muscle_activations.index[0]
        self.duration = self.end_time - self.start_time
        self.num_frames = self.muscle_activations.shape[0]
        self.fps = round(self.num_frames / self.duration, 0)

    def get_muscle_activations(
        self,
        time_window: Tuple[float, float],
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
        np.ndarray: The resampled values as a numpy array
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
        time_window: Tuple[float, float],
        target_fps=20.0,
        rolling_average=False,
        target_frame_count=64,
        as_numpy=False,
    ):
        """
        Resample grf to the given time window and fps. Returns the values as a numpy array or a dataframe

        Parameters:
        time_window (Tuple[float, float]): The start and end times for the window
        target_fps (float): The target frames per second for resampling
        rolling_average (bool): Whether to apply a rolling average

        Returns:
        np.ndarray: The resampled values as a numpy array
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
        time_window: Tuple[float, float],
        target_fps=20.0,
        rolling_average=False,
        target_frame_count=64,
        as_numpy=False,
    ):
        """
        Resample forces to the given time window and fps. Returns the values as a numpy array or a dataframe

        Parameters:
        time_window (Tuple[float, float]): The start and end times for the window
        target_fps (float): The target frames per second for resampling
        rolling_average (bool): Whether to apply a rolling average

        Returns:
        np.ndarray: The resampled values as a numpy array
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


class MintDataset(data.Dataset):
    """
    A PyTorch dataset for the MINT dataset

    Parameters:
    dataset_path (str): The path to the dataset
    transform (Optional[Callable]): A transform to apply to the data
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.metadata = concatenate_mint_metadata(dataset_path)

    def __len__(self):
        # return the number of samples of the dataframe (metadata)
        return self.metadata.shape[0]

    def __getitem__(self, idx: int) -> MintData:
        """Get a sample by its index"""
        sample = self.metadata.iloc[idx]

        muscle_activations = load_pkl_file(
            self.dataset_path, sample["data_path"], "muscle_activations"
        )
        grf = load_pkl_file(self.dataset_path, sample["data_path"], "grf")
        forces = load_pkl_file(self.dataset_path, sample["data_path"], "forces")

        return MintData(muscle_activations, grf, forces)

    def by_path_id(self, path_id: str):
        """
        Get a sample by its path_id

        Parameters:
        path_id (int): The path_id of the sample

        Returns:
        dict: The sample
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
        dict: The sample
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
        dict: The sample
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
        dict: The sample
        """
        filtered = self.metadata[self.metadata["subject"] == subject]
        filtered = filtered[filtered["sequence"] == sequence]
        if filtered.empty:
            raise ValueError(
                f"No sample found with subject: {subject} and sequence: {sequence}"
            )
        idx = self.metadata.index.get_loc(filtered.index[0])
        return self[idx]


if __name__ == "__main__":
    """ Example usage of the MintDataset class """
    dataset_path = "/lsdf/data/activity/MuscleSim/musclesim_dataset"
    mint_dataset = MintDataset(dataset_path)
    sample = mint_dataset[0]
    print(sample.get_forces(None))
    sample_by_path_id = mint_dataset.by_path_id("s1/acting2")
    print(sample_by_path_id.fps)
    sample_by_path = mint_dataset.by_path("TotalCapture/TotalCapture/s1/acting2_poses")
    print(sample_by_path.get_forces(None))
    sample_by_babel_sid = mint_dataset.by_babel_sid(12906)
    print(sample_by_babel_sid.get_forces(None))
    sample_by_subject_and_sequence = mint_dataset.by_subject_and_sequence(
        "s1", "acting2_poses"
    )
