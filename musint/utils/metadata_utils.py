import pandas as pd
import os
import os.path as osp

filenames = {
    "muscle_activations": "muscle_activations.pkl",
    "grf": "grf.pkl",
    "forces": "muscle_forces.pkl",
}

def concatenate_mint_metadata(dataset_path: str, fillna: bool = True, delete_old: bool = False):
    """
    Returns a concatenated dataframe of all the CSV files in the dataset_path.
    If the concatenated dataframe has already been created, it will be read from the output CSV file.

    Parameters:
    dataset_path (str): The path to the mint dataset
    """
    output_file = osp.join(dataset_path, "muscle_activations.csv")

    if not osp.exists(output_file) or delete_old:
        df_list = []

        # Walk through all directories and files in the dataset_path
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".csv"):
                    # Construct the full file path
                    file_path = os.path.join(root, file)
                    # Read the CSV file into a dataframe
                    df = pd.read_csv(file_path)
                    # Append the dataframe to the list
                    df_list.append(df)

        # Concatenate all dataframes in the list into a single dataframe
        df_concat = pd.concat(df_list, ignore_index=True)

        df_concat = convert_types_mint_metadata(df_concat, fillna)

        df_concat = append_path_id_to_mint_metadata(df_concat)

        # Write the concatenated dataframe to a new CSV file
        df_concat.to_csv(output_file, index=False)

    else:
        df_concat = pd.read_csv(output_file)

        df_concat = convert_types_mint_metadata(df_concat, fillna)

    df_concat.set_index("path_id", inplace=True)

    return df_concat


def convert_types_mint_metadata(df: pd.DataFrame, fillna: bool = True):
    """
    Converts the columns of the dataframe to the correct data types.

    Parameters:
    df (pd.DataFrame): The input dataframe from the mint dataset
    """
    df = df.astype(str)  # Change all other columns to str
    df[["height_cm", "weight_kg", "babel_sid", "amass_dur", "analysed_dur", "analysed_%"]] = df[
        ["height_cm", "weight_kg", "babel_sid", "amass_dur", "analysed_dur", "analysed_%"]
    ].astype(float)
    df["babel_sid"] = df["babel_sid"].fillna(-1)
    df[["babel_sid"]] = df[["babel_sid"]].astype(int)

    return df


def append_path_id_to_mint_metadata(df: pd.DataFrame):
    """
    Appends a new column to the dataframe with the last two substrings of the data_path.

    Parameters:
    df (pd.DataFrame): The input dataframe from the mint dataset
    """
    df["path_id"] = df["data_path"].str.split("/").str[-2:].str.join("/").str.replace("_poses", "")
    return df


def get_pkl_file_path(dataset_path: str, data_path: str, file_type: str = "muscle_activations"):
    """
    Returns the full file path of the pkl file for the given data_path and filename.

    Parameters:
    dataset_path (str): The path to the mint dataset
    path_id (str): The path_id of the file
    filename (str): The name of the pkl file
    """
    return osp.join(dataset_path, data_path, filenames[file_type])