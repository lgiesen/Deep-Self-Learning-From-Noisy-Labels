[![Python 3.10](https://img.shields.io/badge/python-3.10.8-blue)](https://www.python.org/downloads/release/python-31013/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
<!-- ![GitHub version](https://img.shields.io/github/v/release/lgiesen/Deep-Self-Learning-From-Noisy-Labels?color=green&include_prereleases) -->
# Deep-Self-Learning-From-Noisy-Labels

## Contextualization

This repository contains the code but not the data for a seminar thesis in the module Advanced Topics in Machine Learning conducted at the University of Münster in the summer semester of 2024.
This seminar thesis verifies the Self-Learning with Multi-Prototypes (SMP) approach from the paper [Deep Self-Learning From Noisy Labels](https://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Deep_Self-Learning_From_Noisy_Labels_ICCV_2019_paper.pdf)" by Han et al. (2019).

## Objectives and Key Results

## Repository Structure

- **example_data**: Excerpt of the dataset

## Files Description

<!-- - **.env**: Reused local variables specific to your OS -->
- **config.py**: Reused global variables
<!-- - **requirements.txt**: Lists all Python libraries required to run the scripts -->
- **CreateDataset.py**: Create the dataset
- **LoadDataset.py**: Load the dataset
- **DataExploration.ipynb**: Explore the original data and dataset
<!-- The main script orchestrates the data collection, preprocessing, sentiment analysis, and visualization processes. -->
        
## License

This project is licensed under the [MIT LICENSE](https://github.com/lgiesen/Deep-Self-Learning-From-Noisy-Labels/blob/main/LICENSE) file for details.

## Usage

Prerequisite: You require the data from [Tong](mailto:tong.xiao.work@gmail.com) (see [Noisy Label Repository](https://github.com/Cysu/noisy_label).
1. Adjust the `config.py` file to match the filepath to the data and root directory.
2. Execute the code from `CreateDataset.py` to generate the dataset.
🚧 The final steps will be added upon final implementation.