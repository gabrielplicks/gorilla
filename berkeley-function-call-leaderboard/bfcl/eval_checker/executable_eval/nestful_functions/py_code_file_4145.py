import pandas as pd



def sort_time_series(series: pd.Series) -> list:

    """

    Sorts the provided time series by its index in descending order and returns its values as a list.

    Args:

        series: The time series to be sorted.

    """

    sorted_series = series.sort_index(ascending=False)

    return sorted_series.tolist()

