# Muscles in Time (MinT) Dataset

The `musint` package provided in this repository is a Python companion package for the Muscles in Time (MinT) dataset. This dataset is a large-scale synthetic muscle activation dataset, derived from biomechanically accurate simulations using OpenSim, and is designed to advance research in human motion understanding. This package facilitates access to the MinT dataset, providing tools for efficient data handling, preprocessing, and integration with existing human motion datasets.

## Overview

The Muscles in Time project introduces a comprehensive dataset that bridges the gap between surface-level motion data and the underlying muscle activations that drive these motions. The MinT dataset, built on top of existing motion capture datasets, includes over 9 hours of simulated muscle activation data covering 227 subjects and 402 muscle strands. This dataset supports the exploration of the complex dynamics of human motion and is particularly valuable for training neural networks in biomechanical and computer vision research.

## Key Features

- **Comprehensive Muscle Activation Data**: Includes detailed muscle activation sequences for a wide range of human motions, simulated using OpenSim's validated biomechanical models.
- **Cross-Compatibility**: Facilitates integration with the BABEL dataset, AMASS, and other motion capture datasets, enhancing the usability of the dataset across different research projects.
- **Preprocessing Utilities**: Provides tools for segmenting data, handling missing values, and converting between different time frames, making the dataset easy to work with.
- **Flexible Data Handling**: Includes utilities for loading data as pandas DataFrames or PyTorch datasets, enabling seamless integration with existing machine learning workflows.

## Repository Structure

```sh
└── ./
    ├── LICENSE
    ├── MANIFEST.in
    ├── README.md
    ├── musint
    │   ├── __init__.py
    │   ├── benchmarks
    │   ├── datasets
    │   ├── muscle_groups
    │   ├── tests
    │   └── utils
    └── pyproject.toml
```

## Modules

### `musint.datasets`
This module includes classes and methods for loading and managing the MinT dataset. It provides functionality for accessing muscle activation data, ground reaction forces, and motion data in a structured format.

### `musint.benchmarks`
Contains predefined subsets of muscle groups based on established biomechanical models. These subsets are useful for benchmarking and evaluating the performance of neural networks in predicting muscle activations.

### `musint.utils`
Utility functions for data preprocessing, including methods for resampling muscle activation data, converting between time and frame indices, and handling metadata.

### `musint.muscle_groups`
Defines the muscle groups used in the dataset, categorized into lower body and thoracolumbar regions. These definitions are critical for understanding the anatomical basis of the dataset.

## Getting Started

### Installation

To install the `musint` package, use pip:

```bash
pip install musint
```

### Usage

To load and interact with the MinT dataset, follow this example:

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

### Data Structure

The dataset is provided in CSV files or pandas DataFrames stored in pickle files. Each file contains muscle activation data indexed by fractional timestamps, with columns representing different muscle strands.

## Project Roadmap

Future releases will include:
- Public access to the full MinT dataset via a DOI for long-term storage and citation.
- Additional tools for data augmentation and advanced preprocessing.
- Benchmarks and comparison studies with real-world datasets.


## License

The Muscles in Time dataset and the `musint` package are released under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/). The code for the data generation pipeline is licensed under the [Apache License 2.0](https://apache.org/licenses/LICENSE-2.0).

## Acknowledgments

This work was supported by the Karlsruhe Institute of Technology. The dataset is based on contributions from various publicly available datasets, including AMASS, BMLmovi, and KIT Whole-Body Human Motion Database.
```
