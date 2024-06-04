from typing import Tuple

import pandas as pd
import numpy as np

EPOCH = pd.Timestamp("1970-01-01")


def trim_mint_dataframe(
    df: pd.DataFrame,
    time_window: Tuple[float, float],
    target_fps=20.0,
    rolling_average=False,
    target_frame_count=64,
    as_numpy=True,
    max_frame_gap=(1 / 50) + 1e-2,  # This is the expected gap given that the MinT dataset is sampled at 50Hz.
):
    """
    Resample muscle activations to the given time window and fps. Returns the values as a numpy array

    Parameters:
    df (pd.DataFrame): The input dataframe from the mint dataset
    time_window (Tuple[float, float]): The start and end times for the window
    target_fps (float): The target frames per second for resampling
    rolling_average (bool): Whether to apply a rolling average

    Returns:
    np.ndarray: The resampled values as a numpy array or a dataframe
    """
    # ms of resampling depending on the fps
    resampling_ms = 1000 / target_fps

    # trim the dataframe to the time window
    filtered_df = df[df.index.to_series().between(time_window[0], time_window[1], inclusive="both")]

    # check for a gap in the indices which resemble float timestamps
    differences = filtered_df.index.to_series().diff()
    if any(differences > max_frame_gap):  # This is the expected gap given that the MinT dataset is sampled at 50Hz.
        raise ValueError(f"Found a gap in the timestamps larger than {max_frame_gap:.2f}s")

    # convert to datetime
    filtered_df.index = pd.to_datetime(filtered_df.index, unit="s")

    if rolling_average:
        # rolling average with 50ms window
        filtered_df = filtered_df.rolling(f"{resampling_ms}ms").mean()

    # resample to 50ms
    filtered_df = filtered_df.resample(f"{resampling_ms}ms").nearest()

    # convert back to timestamp
    filtered_df.index = (filtered_df.index - EPOCH) / pd.Timedelta("1s")

    if target_frame_count is not None:
        # trim to target_frame_count
        filtered_df = filtered_df.head(target_frame_count)

    if as_numpy:
        return filtered_df.values
    else:
        return filtered_df


def trim_mint_dataframe_v2(
    df: pd.DataFrame,
    time_window: Tuple[float, float],
    target_frame_count=None,
    as_numpy=True,
):
    """
    Trims and resamples a DataFrame from the MinT dataset to a specified time window and frame count.
    Optionally returns the result as a NumPy array.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing muscle activation data from the MinT dataset.
    - time_window (Tuple[float, float]): Tuple specifying the start and end times of the desired time window.
    - target_frame_count (int, optional): The number of frames to resample the data to within the time window. Default is 64.
    - as_numpy (bool, optional): If True, the resulting resampled data is returned as a NumPy array; otherwise, as a DataFrame. Default is True.

    Returns:
    - If as_numpy is True: NumPy array containing the resampled muscle activation data.
    - If as_numpy is False: DataFrame containing the resampled muscle activation data.
    """

    filt_df = df[(df.index >= time_window[0]) & (df.index <= time_window[1])]

    if target_frame_count is None:
        return filt_df.values if as_numpy else filt_df

    # Create a grid of target time points
    grid = np.linspace(start=time_window[0], stop=time_window[1], num=target_frame_count)

    # Compute the absolute differences between each grid point and all index values in a vectorized manner
    abs_diff_matrix = np.abs(filt_df.index.values[:, np.newaxis] - grid)

    # Find the index of the closest time point for each grid point
    closest_indices = abs_diff_matrix.argmin(axis=0)

    # Select the closest rows based on the indices, ensuring each row is selected only once
    unique_indices = closest_indices
    closest_rows = filt_df.iloc[unique_indices]

    return closest_rows.values if as_numpy else closest_rows


def frame_to_time(frame: int, fps: float):
    """
    Converts a frame number to a time in seconds
    """
    return round(frame / fps, 2)


def time_to_frame(time: float, fps: float):
    """
    Converts a time in seconds to a frame number
    """
    return int(time * fps)
