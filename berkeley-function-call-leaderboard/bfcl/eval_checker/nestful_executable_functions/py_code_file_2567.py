import pandas as pd



def normalize_and_rescale(dataset: pd.DataFrame) -> pd.DataFrame:

    """Normalizes and rescales a time series dataset.

    Args:

        dataset: A pandas DataFrame containing columns 'Time', 'Value', and 'Diff'.

    """

    dataset['Normalized Value'] = dataset['Value'] / dataset['Value'].max() * 100

    dataset['Rescaled Time'] = dataset['Time'] - dataset['Time'].min()

    return dataset

