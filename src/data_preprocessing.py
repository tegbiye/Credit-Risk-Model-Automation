import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt


def data_loader(filepath):
    """
    Load a CSV file into a DataFrame.

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"CSV file loaded successfully from {filepath}.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file from {filepath}: {e}")
        return None


def plot_feature_distribution(data_series, title, xlabel, xlim=None):
    plt.figure(figsize=(12, 6))
    sns.histplot(data_series, bins=50, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    if xlim:
        plt.xlim(xlim)
    plt.show()


def plot_categorical_distribution(data, title, xlabel, order=None):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=xlabel, order=data[xlabel].value_counts().index)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
