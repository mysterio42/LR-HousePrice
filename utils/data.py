import numpy as np
import pandas as pd
from pandas import DataFrame


def load_data(path: str) -> DataFrame:
    """
    :param path: Relative path of csv file
    :return: DataFrame of the Source Data
    """
    return pd.read_csv(path, header=0, encoding='unicode_escape')


def prepare_data(data: DataFrame, features: tuple) -> tuple:
    """
    :param data: Data Source pandas DataFrame
    :param features: Column names form tuple ('price','sqft_living','floors') etc.
    :return: tuple of DataFrames formed appropriate features
    """
    return tuple(data[feature] for feature in features)


def to_array(data: tuple):
    """

    :param data: tuple of appropriate features  DataFrame
    :return: tuple of appropriate features      numpy.ndarray
    """
    return tuple(np.array(feature, dtype=np.float).reshape((-1, 1)) for feature in data)
