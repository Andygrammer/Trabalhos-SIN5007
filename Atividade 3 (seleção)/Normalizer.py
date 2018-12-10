import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler


def standardize(df):
    """
    normalizacao
    :param df:
    :return:
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def normalize_by_zscore(dataset: pd.DataFrame, list_of_columns) -> pd.DataFrame:
    """
    UNUSED
    Funcao para normalizacao do dataset usando z-score.
    Esta funcao usa a implementacao do z-score da biblioteca scipy.
    Tem variacao minuscula do calculo manual (provavelmente devido a precisao).

    :param dataset:
    :param list_of_columns:
    :return:
    """
    for column in dataset:
        if column in list_of_columns:
            dataset[column] = dataset[column].pipe(zscore, ddof=1)
    return dataset

def normalize_by_manual_zscore(dataset: pd.DataFrame, list_of_columns:[]) -> pd.DataFrame:
    """
    UNUSED
    Funcao para normalizacao do dataset usando zscore, utilizando implementacao propria.

    :param dataset:
    :param list_of_columns:
    :return:
    """

    for column in dataset:
        if column in list_of_columns:
            dataset[column] = dataset[column].astype(float)
            mean = dataset[column].mean()
            std = dataset[column].std(ddof=1)

            dataset[column] = manual_zscore(dataset[column], mean, std)

    return dataset


def manual_zscore(series, mean, stdev):
    """UNUSED"""
    series = (series - mean)/stdev
    return series


