from typing import List

import pandas as pd



def sort_and_return(series: pd.Series, ascending: bool) -> List[float]:

    """Sorts a Pandas Series and returns the first or last 5 elements as a list.



    Args:

        series: The Pandas Series to be sorted.

        ascending: If True, sorts the Series in ascending order and returns the first 5 elements.

            If False, sorts the Series in descending order and returns the last 5 elements.



    Returns:

        A list of the first or last 5 elements of the sorted Series.

    """

    sorted_series = series.sort_values(ascending=ascending)

    if ascending:

        return sorted_series.head().tolist()

    else:

        return sorted_series.tail().tolist()

