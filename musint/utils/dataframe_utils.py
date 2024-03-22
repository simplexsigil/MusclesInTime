import pandas as pd
from typing import Tuple

EPOCH = pd.Timestamp("1970-01-01")

def trim_mint_dataframe(df: pd.DataFrame, time_window: Tuple[float, float], target_fps=20.0, rolling_average=False, target_frame_count=64, as_numpy=True, check_gaps=False):
    '''
    Resample muscle activations to the given time window and fps. Returns the values as a numpy array

    Parameters:
    df (pd.DataFrame): The input dataframe from the mint dataset
    time_window (Tuple[float, float]): The start and end times for the window
    target_fps (float): The target frames per second for resampling
    rolling_average (bool): Whether to apply a rolling average

    Returns:
    np.ndarray: The resampled values as a numpy array or a dataframe
    '''
    # ms of resampling depending on the fps
    resampling_ms = 1000 / target_fps

    # trim the dataframe to the time window
    filtered_df = df[df.index.to_series().between(time_window[0], time_window[1], inclusive='both')]

    # check for a gap in the indices which resemble float timestamps
    if check_gaps:
        differences = filtered_df.index.to_series().diff()
        if any(differences > 0.021):
            raise ValueError("Found a gap in the timestamps larger than 0.02s")

    # convert to datetime
    filtered_df.index = pd.to_datetime(filtered_df.index, unit='s')

    if rolling_average:
        # rolling average with 50ms window
        filtered_df = filtered_df.rolling(f'{resampling_ms}ms').mean()

    # resample to 50ms
    filtered_df = filtered_df.resample(f'{resampling_ms}ms').nearest()

    # convert back to timestamp
    filtered_df.index = (filtered_df.index - EPOCH) / pd.Timedelta('1s')

    if target_frame_count is not None:
        # trim to target_frame_count
        filtered_df = filtered_df.head(target_frame_count)

    if as_numpy:
        return filtered_df.values
    else:
        return filtered_df
    

def frame_to_time(frame: int, fps: float):
    """
    Converts a frame number to a time in seconds
    """
    return frame / fps

def time_to_frame(time: float, fps: float):
    """
    Converts a time in seconds to a frame number
    """
    return time * fps