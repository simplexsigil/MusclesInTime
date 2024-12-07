# Muscles in Time (MinT) Dataset

The `musint` package in this repository is a Python toolset for the Muscles in Time (MinT) dataset, a large-scale synthetic muscle activation dataset derived from biomechanically accurate OpenSim simulations. This dataset bridges surface-level motion data with the underlying muscle activations.

For more information also have a look at our website:
  - [Muscles in Time Data](https://simplexsigil.github.io/mint)

The actual data can be downloaded from here:
 - [Muscles in Time Data](https://radar.kit.edu/radar/en/dataset/VDPCEFSThBWlDPFL.Muscles%2BTime)

Note that the data release server zips additional metadata together with the compressed MinT dataset which means you have to extract this outer container first. The actual compressed MinT data can be found within under:
`10.35097-VDPCEFSThBWlDPFL\10.35097-VDPCEFSThBWlDPFL\data\dataset\MinT.tar.zst`

It requires Zstandard for decompression. Modern versions of `tar` should support it out-of-the-box, for older versions you might have to install `zstd` and pass it to tar via the `-I zstd` option.

We also performed some analysis, checking various metrics to get a better understanding of simulation quality. This data is provided both in human readable (text log file and pdf) and machine readable form (pickle file) as metadata. Please have a look at our website to find the link to that metadata. It is not required to use MinT, unless you want to filter samples based on these quality metrics.

The MinT dataset offers over 9 hours of simulated muscle activation data, covering 227 subjects and 402 muscle strands. 
It is built by performing OpenSim simulation on the following motion capture datasets:
- [BML-MoVi and BML-RUB](https://www.biomotionlab.ca/movi/)
- [KIT Whole-Body Human Motion Database](https://motion-database.humanoids.kit.edu/)
- [EyesJapan Dataset](http://mocapdata.com/Terms_of_Use.html)
- [Total Capture](https://cvssp.org/data/totalcapture/)

For the generation of MinT we made use of AMASS by placing virtual markers on the SMPL body model in order to use them as cross-dataset normalized input markers for OpenSim.

- [AMASS](https://amass.is.tue.mpg.de/index.html)

### Key Features

1. **Comprehensive Muscle Activation Data**:
   - The dataset includes detailed muscle activation sequences for a wide range of human motions.
   - These sequences are simulated using validated biomechanical OpenSim models.

2. **Multi-modality and cross-dataset compatibility**:
   - The dataset is designed to be compatible with the motion capture dataset AMASS and its accompanying textual description dataset BABEL.
   - This enhances the usability of the dataset for multi-modality research projects.

3. **Preprocessing Utilities**:
   - The project provides tools for segmenting data, handling missing values, and converting between different time frames and frame frequencies.
   - These utilities make the dataset easier to work with and integrate into existing workflows.

4. **Flexible Data Handling**:
   - Utilities are included for loading data as pandas DataFrames or PyTorch datasets.
   - This flexibility enables seamless integration with existing machine learning workflows, facilitating the training of neural networks in biomechanical and computer vision research.

### Getting Started

#### Installation

To install the `musint` package, use pip:

```bash
pip install musint
```


#### Example Usages
Here are examples of how to use the `MintDataset` class:
```python
import argparse
from musint.datasets.mint_dataset import MintDataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True)
dataset_path = parser.parse_args().dataset_path

mint_dataset = MintDataset(dataset_path, use_cache=True)

# Retrieve sample by path
sample_by_path = mint_dataset.by_path("TotalCapture/TotalCapture/s1/acting2_poses")
print(sample_by_path)

# Retrieve sample by path ID
path_id = mint_dataset.path_ids[0]
print(f"Path ID: {path_id}")

sample_by_path_id = mint_dataset.by_path_id(path_id)

# Retrieve sample by BABEL SID
sample_by_babel_sid = mint_dataset.by_babel_sid(12906)

# Retrieve sample by HumanML3D name
sample_by_humanml3d_name = mint_dataset.by_humanml3d_name("003260")[0]

# Get valid indices
valid_indices = sample_by_humanml3d_name.get_valid_indices((10, 1000), 20.0)

# Get gaps in data
gaps = mint_dataset.get_gaps(as_frame=True, target_fps=20.0)
print(gaps)
```

```python
import os
from musint.datasets.mint_dataset import MintDataset

# Load the dataset
dataset = MintDataset(os.path.expandvars("$MINT_ROOT"))

# Access a specific sequence
data = dataset.by_path("TotalCapture/TotalCapture/s1/acting2_poses")

# Get muscle activations for a specific time window
activations = data.get_muscle_activations(time_window=(0.3, 1.0), target_frame_count=14)
print(activations)
```

### Repository Structure

```sh
.
└── musint
    ├── benchmarks
    │   └── muscle_sets.py
    ├── datasets
    │   ├── amass_dataset.py
    │   ├── babel_dataset.py
    │   ├── humanml3d_index.csv
    │   └── mint_dataset.py
    ├── muscle_groups
    │   ├── lu_muscle_groups.py
    │   └── tl_muscle_groups.py
    └── utils
        ├── dataframe_utils.py
        ├── hml3d_utils
        │   ├── common
        │   │   ├── quaternion.py
        │   │   └── skeleton.py
        │   ├── humanml3d_utils.py
        │   └── paramUtil.py
        └── metadata_utils.py
```

### Subpackages

The repository is organized into several subpackages, each serving a specific purpose:

**[`musint.datasets`](musint/datasets)**
  - Contains classes and methods for loading and managing the MinT dataset as well as AMASS and BABEL.
  - Provides functionality for accessing muscle activation data, ground reaction forces, and motion data in a structured format.

**[`musint.benchmarks`](musint/benchmarks)**
  - Includes predefined subsets of muscle groups based on established biomechanical models.
  - Useful for benchmarking and evaluating the performance of neural networks in predicting muscle activations.

**[`musint.utils`](musint/utils)**
  - Utility functions for data preprocessing, including methods for resampling muscle activation data, converting between time and frame indices, and handling metadata.

**[`musint.muscle_groups`](musint/muscle_groups)**
  - Defines the muscle groups used in the dataset, categorized into lower body and thoracolumbar regions.
  - These definitions are critical for understanding the anatomical basis of the dataset.


#### Classes
The intuition behind `musint` package is to provide a structured and efficient way to work with the MinT dataset. By encapsulating the dataset in a class, it allows for:
- **Efficient Data Access**: Methods to retrieve data by different identifiers used in AMASS, BABEL or HumanML3D (e.g., subject, sequence, path ID).
- **Data Manipulation**: Functions to generate sub segments, convert between fps, and handle gaps in data.
- **Caching and Memory Management**: Options to cache data in memory and manage memory usage.

##### [`MintDataset`](musint/datasets/mint_dataset.py#:~:text=class%20MintDataset)
`MintDataset` is a torch dataset class designed to manage and interact with the MinT dataset. It provides methods to load, access, and manipulate the data efficiently. The class handles various indexing and retrieval mechanisms to facilitate easy access to specific data samples based on different criteria.

##### [`MintData`](musint/datasets/mint_dataset.py#:~:text=class%20MintData)
`MintData` is a data sample class that represents individual data records within the MinT dataset, a record being a complete sequence as found in the AMASS dataset. It encapsulates muscle activation, ground reaction force and muscle force data and metadata for a single sample, and relevant meta data.

#### Important Functions

#### [`MintDataset`](musint/datasets/mint_dataset.py#:~:text=class%20MintDataset)
 
```python
def __init__(self, dataset_path, use_cache=True, keep_in_memory=False, pyarrow=False, load_humanml3d_names=True):
```
- **Purpose**: Initializes the `MintDataset` object with the given dataset path and optional parameters.
- **Parameters**:
  - `dataset_path`: Path to the dataset.
  - `use_cache`: Whether to cache precomputed metadata.
  - `keep_in_memory`: Whether to keep the dataset in memory.
  - `pyarrow`: Whether to use PyArrow for data handling.
  - `load_humanml3d_names`: Whether to load HumanML3D names.

##### [`by_subject_and_sequence`](musint/datasets/mint_dataset.py#:~:text=def%20by_subject_and_sequence)
```python
def by_subject_and_sequence(self, subject, sequence):
```
- **Purpose**: Retrieves a sample by subject and sequence.
- **Parameters**:
  - `subject`: The subject identifier.
  - `sequence`: The sequence identifier.
- **Usage**:
  ```python
  sample = mint_dataset.by_subject_and_sequence(subject, sequence)
  ```

##### [`by_segment_name`](musint/datasets/mint_dataset.py#:~:text=def%20by_segment_name)
```python
def by_segment_name(self, path):
```
- **Purpose**: Retrieves a sample based on the segment name.
- **Parameters**:
  - `path`: The segment name.
- **Usage**:
  ```python
  sample = mint_dataset.by_segment_name(path)
  ```

##### [`by_path`](musint/datasets/mint_dataset.py#:~:text=def%20by_path)
```python
def by_path(self, sample_path):
```
- **Purpose**: Retrieves a sample using a path-like index´.
- **Parameters**:
  - `sample_path`: The path-like index.
- **Usage**:
  ```python
  sample = mint_dataset.by_path(sample_path)
  ```

##### [`get_index_by_path_id`](musint/datasets/mint_dataset.py#:~:text=def%20get_index_by_path_id)
```python
def get_index_by_path_id(self, path_id):
```
- **Purpose**: Retrieves the index of a sample by its path ID.
- **Parameters**:
  - `path_id`: The path ID.
- **Usage**:
  ```python
  index = mint_dataset.get_index_by_path_id(path_id)
  ```

##### [`by_path_id`](musint/datasets/mint_dataset.py#:~:text=def%20by_path_id)
```python
def by_path_id(self, path_id):
```
- **Purpose**: Retrieves a sample by its path ID.
- **Parameters**:
  - `path_id`: The path ID.
- **Usage**:
  ```python
  sample = mint_dataset.by_path_id(path_id)
  ```

##### [`by_babel_sid`](musint/datasets/mint_dataset.py#:~:text=def%20by_babel_sid)
```python
def by_babel_sid(self, babel_sid):
```
- **Purpose**: Retrieves a sample by its BABEL SID.
- **Parameters**:
  - `babel_sid`: The BABEL SID.
- **Usage**:
  ```python
  sample = mint_dataset.by_babel_sid(babel_sid)
  ```

##### [`by_humanml3d_name`](musint/datasets/mint_dataset.py#:~:text=def%20by_humanml3d_name)
```python
def by_humanml3d_name(self, humanml3d_name):
```
- **Purpose**: Retrieves a sample by its HumanML3D name.
- **Parameters**:
  - `humanml3d_name`: The HumanML3D name.
- **Usage**:
  ```python
  sample = mint_dataset.by_humanml3d_name(humanml3d_name)
  ```

#### [`MintData`](musint/datasets/mint_dataset.py#:~:text=class%20MintData)

##### [`get_gaps`](musint/datasets/mint_dataset.py#:~:text=def%20get_gaps)
```python
def get_gaps(self, as_frame=False, target_fps=20.0):
```
- **Purpose**: Identifies gaps in the data.
- **Parameters**:
  - `as_frame`: Whether to return gaps as frame numbers or as timestamps.
  - `target_fps`: Target frames per second.
- **Usage**:
  ```python
  gaps = mint_dataset.get_gaps(as_frame=True, target_fps=20.0)
  ```


#### Utils and other functions

##### [trim_mint_dataframe_v2](musint/utils/processing.py#:~:text=def%20trim_mint_dataframe_v2)
```python
def trim_mint_dataframe_v2(
    df: pd.DataFrame,
    time_window: Tuple[float, float],
    target_frame_count=None,
    as_numpy=True,
):
```

- **Purpose**: Trims and resamples a DataFrame from the MinT dataset to a specified time window and frame count. Optionally returns the result as a NumPy array. Vectorized version of this function for faster computation. Note, that fps is implicit by setting time_window and target_frame_count accordingly.
- **Parameters**:
  - `df` (pd.DataFrame): Input DataFrame containing muscle activation data from the MinT dataset.
  - `time_window` (Tuple[float, float]): Tuple specifying the start and end times of the desired time window.
  - `target_frame_count` (int, optional): The number of frames to resample the data to within the time window. Default is 64.
  - `as_numpy` (bool, optional): If True, the resulting resampled data is returned as a NumPy array; otherwise, as a DataFrame. Default is True.
- **Returns**:
  - If `as_numpy` is True: NumPy array containing the resampled muscle activation data.
  - If `as_numpy` is False: DataFrame containing the resampled muscle activation data.
- **Usage**:
  
```python
trimmed_data = trim_mint_dataframe_v2(df, time_window=(0.0, 5.0), target_frame_count=64, as_numpy=True)
```

#### Input Data Structure

The structure of MinT data is intentionally kept simple. All data is saved in CSV files or pandas DataFrames stored in pickle files, the data is provided with 50 fps, each dataframe is indexed by fractional timestamps. The `musint` package loads these files, makes them indexable while cross-referencing them to other datasets like AMASS and BABEL and allows for loading subsegments at varying target frame rates. The structure of the data is otherwise maintained by `musint`.

Columns are named meaningfully, the first 80 muscles belong to the lower body model, the following 322 muscels belong to the upper body model. The ordered column names of these dataframes are also provided in [`musint/benchmarks/muscle_sets.py`](musint/benchmarks/muscle_sets.py) in [`MUSCLE_SUBSETS[MUSINT_402]`](musint/benchmarks/muscle_sets.py#:~:text=MUSINT_402). [`musint/benchmarks/muscle_sets.py`](musint/benchmarks/muscle_sets.py) also contains meaningful muscle subsets which can be used for benchmarking. 

The first and last 0.14 seconds are cut off since the muscle activation analysis is unstable towards the beginning and end of data, so the timestamp index of a dataframe starts at 0.14 seconds. Since the data is generated in chunks of $1.4$ seconds and muscle activation analysis can fail to succeed due to various factors, the provided concatenated data may contain gaps identified by missing dataframe rows for such time ranges.

In the following we show how data can be loaded **without** using the musint package to explore the data without any processing.

```python
>>> # First download and extract the dataset.
>>> # Example for sample 
>>> #'BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_10_poses'
>>> import joblib
>>> joblib.load("muscle_activations.pkl")
      LU_addbrev_l    ...   TL_TR4_r  TL_TR5_r
0.14         0.016    ...      0.003     0.061
0.16         0.028    ...      0.005     0.070
0.18         0.033    ...      0.002     0.080
...            ...    ...        ...       ...
3.74         0.024    ...      0.020     0.028
3.76         0.016    ...      0.009     0.004
3.78         0.011    ...      0.003     0.000

[183 rows x 402 columns]

>>> joblib.load("grf.pkl")
      ground_force_right_vx  ...  ground_torque_left_z
0.14                 15.962  ...                   0.0
0.16                 10.596  ...                   0.0
0.18                  3.422  ...                   0.0
...                     ...  ...                   ...
3.72                 20.337  ...                   0.0
3.74                 21.572  ...                   0.0
3.76                 22.546  ...                   0.0

[182 rows x 18 columns]

>>> joblib.load("muscle_forces.pkl")
      LU_addbrev_l   ...  TL_TR4_r  TL_TR5_r
0.14         8.430   ...     0.153    11.652
0.16        15.345   ...     0.283    13.240
0.18        19.127   ...     0.143    15.240
...            ...   ...       ...       ...
3.72        14.437   ...     1.320     3.661
3.74        13.993   ...     1.270     5.330
3.76         9.346   ...     0.577     0.847

[182 rows x 402 columns]
```

### Project Roadmap

Future releases will include:
- Public access to the full MinT dataset via a DOI for long-term storage and citation.
- Additional tools for data augmentation and advanced preprocessing.


### License

The Muscles in Time dataset and the `musint` package are released under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/). The code for the data generation pipeline is licensed under the [Apache License 2.0](https://apache.org/licenses/LICENSE-2.0).

### Acknowledgments

This work was supported by the Karlsruhe Institute of Technology. The dataset is based on contributions from various publicly available datasets, including AMASS, BMLmovi, and KIT Whole-Body Human Motion Database.

