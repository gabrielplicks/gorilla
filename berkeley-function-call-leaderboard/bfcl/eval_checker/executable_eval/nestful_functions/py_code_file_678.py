from typing import List

import pandas as pd



def sort_values_column(df: pd.DataFrame) -> List[str]:

    """Returns a sorted list of the values in the `z` column of a data frame `df`.



    Args:

        df: The input data frame containing the `z` column.



    Returns:

        A sorted list of the values in the `z` column.

    """

    return sorted(df['z'])

