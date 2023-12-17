[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/U9FTc9i_)

# Machine Learning Course - Class Project 1 - Fall 2023

## Project Overview

This repository contains code and files for implementing various machine learning methods for the second project of the machine learning class. The project's main goal is to develop and evaluate different machine learning algorithms on a twitter dataset for binary sentiment analysis. This repository contains files to create submissions for AI crowd and to recreate the plos shown in the report.

## Contributors

- Aly Elbindary
- André Schakkal
- Peter Harmouch

Group Name: APASHE

## Files and Structure

The structure of the folder is the following:

```
ML-PROJECT2-2-APASHE/
├── data/
│ ├── test_data.txt
│ ├── train_neg.txt
│ ├── train_neg_full.txt
│ ├── train_pos.txt
│ └── train_pos_full.txt
├── submissions/
├── weights/
├── word_embeddings/
│ ├── cooc_files/
│ ├── encoded_dfs/
│ ├── saved_results/
│ ├── vocab_cut_files/
│ ├── vocab_full_files/
│ ├── vocab_pkl_files/
│ └── word_embeddings_plots.ipynb
├── tf_idf/
├── run.ipynb
├── train.ipynb
├── plots.ipynb
├── helpers.py
├── requirements.txt
└── README.md
```

The most important files are the following:
1. `run.ipynb`: This Jupyter Notebook imports a pretrained model, applies it to the dataset, and creates a CSV file suitable for submission on AI Crowd

2. `train.ipynb`: This Jupyter Notebook is dedicated to training our best working model. It creates a txt file that can then be imported in run.ipynb.

3. `helpers.py`: This Python script contains useful functions used throughout the project.

4. `plots.ipynb`: This Jupyter Notebook is used to generate relevant plots related to the final method for the project report. It includes hyperparameter search plots and plots for an ablation study. The different plots are seperated by markdowns.

## Data

The project data should be put in the folder called `data`. You can download the dataset from the following URL: [ML_course repository on GitHub](https://github.com/epfml/ML_course).


## Running the Code

To run the code in this project, follow these steps:

1. Make sure you have the necessary libraries (numpy, and matplotlib), installed in your Python environment. You can set up a Conda environment with the required libraries using the following steps:

```bash
conda create --name ml-project python=3.8
conda activate ml-project
conda install -r requirements.txt
```

2. If you only want to test our model. We have already generated a ready to use `w.txt` file that contains our pretrained model obtained from ridge regression. You can obtain the csv file by running the file `run.ipynb`, you then have to submit this file on aicrowd to get the accuracy on the testing set. Make sure to run all the blocks in order.

2. If you want to train the model yourself, open and run the `train.ipynb` notebook. Make sure to run all the blocks in order. After training, you can use the pretrained model by opening and running the `run.ipynb` notebook to create a submission file for AI Crowd.

4. Use the `plots.ipynb` notebook to generate relevant plots for your project report, such as hyperparameter search and ablation study plots.
