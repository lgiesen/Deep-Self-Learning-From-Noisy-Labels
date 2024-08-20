[![Python 3.10](https://img.shields.io/badge/python-3.10.8-blue)](https://www.python.org/downloads/release/python-31013/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
<!-- ![GitHub version](https://img.shields.io/github/v/release/lgiesen/Deep-Self-Learning-From-Noisy-Labels?color=green&include_prereleases) -->
# Self-Learning With Multiple Prototypes (SMP) - Deep Self-Learning From Noisy Labels


## Contextualization and Objectives

This repository contains the code for a seminar thesis in the module <i>Advanced Topics in Machine Learning</i> conducted at the University of Münster in the summer semester of 2024.
This seminar thesis evaluats the effectiveness and efficiency of the Self-Learning with Multi-Prototypes (SMP) approach from the paper [Deep Self-Learning From Noisy Labels](https://arxiv.org/abs/1908.02160) ([pdf](https://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Deep_Self-Learning_From_Noisy_Labels_ICCV_2019_paper.pdf)) by Han et al. (2019).

## Repository Structure

- **example_data/**: Excerpt of the dataset
- **model/**: Model results, visualization and trained models
- **notebooks/**: Data exploration, standard and SMP approach code in Jupyter notebook
    - **dataexploration.ipynb**: Explore the original data and dataset
    - **standardapproach.ipynb**: Train and evaluate the model with a standard approach
    - **smpapproach.ipynb**: Train and evaluate the model based on the SMP approach
- **scripts/**: Dataset generation, standard and SMP approach code in Python
    - **createdataset.py**: Create the dataset
    - **loaddataset.py**: Load the dataset
- **thesis_code/**: Extracted SMP code for the seminar thesis
- **.env**: Local runtime-specific variables
- **config.py**: Reused global variables
<!-- - **requirements.txt**: Lists all Python libraries required to run the scripts -->
<!-- The main script orchestrates the data collection, preprocessing, sentiment analysis, and visualization processes. -->
        
## License

This project is licensed under the [MIT LICENSE](https://github.com/lgiesen/Deep-Self-Learning-From-Noisy-Labels/blob/main/LICENSE) file for details.

## Usage

Prerequisite: You require the data from [Tong](mailto:tong.xiao.work@gmail.com) (see [Noisy Label Repository](https://github.com/Cysu/noisy_label).
1. Adjust the `config.py` and `.env` file to adjust the filepaths and parameters to your liking.
2. Install requirements.
    ```
    pip install -r requirements.txt
    ```
3. Execute the code from `scripts/createdataset.py` to generate the dataset.
4. Train and evaluate the models using the notebooks or scripts.
    
    The trained standard model can be found in [model/standard.pth](https://github.com/lgiesen/Deep-Self-Learning-From-Noisy-Labels/blob/main/model/standard.pth)