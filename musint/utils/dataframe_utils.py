import pandas as pd
from typing import Tuple

EPOCH = pd.Timestamp("1970-01-01")

def trim_mint_dataframe(df: pd.DataFrame, time_window: Tuple[float, float], target_fps=20.0, rolling_average=False):
    '''
    Resample muscle activations to the given time window and fps. Returns the values as a numpy array

    Parameters:
    df (pd.DataFrame): The input dataframe from the mint dataset
    time_window (Tuple[float, float]): The start and end times for the window
    target_fps (float): The target frames per second for resampling
    rolling_average (bool): Whether to apply a rolling average

    Returns:
    np.ndarray: The resampled values as a numpy array
    '''
    # ms of resampling depending on the fps
    resampling_ms = 1000 / target_fps

    # trim the dataframe to the time window
    filered_df = df[df.index.to_series().between(time_window[0], time_window[1])]

    # convert to datetime
    filered_df.index = pd.to_datetime(filered_df.index, unit='s')

    if rolling_average:
        # rolling average with 50ms window
        filered_df = filered_df.rolling(f'{resampling_ms}ms').mean()

    # resample to 50ms
    filered_df = filered_df.resample(f'{resampling_ms}ms').nearest()

    # convert back to timestamp
    filered_df.index = (filered_df.index - EPOCH) / pd.Timedelta('1s')

    return filered_df.values