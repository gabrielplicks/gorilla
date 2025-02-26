from typing import Union

import pandas as pd



def get_first_n_rows(n: int, data: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:

    """Extracts the first `n` rows from a DataFrame.



    Args:

        n: The number of rows to extract.

        data: The DataFrame to extract the rows from.



    Returns:

        A DataFrame containing the first `n` rows of the input DataFrame. If the input DataFrame has fewer rows than `n`,

        the function returns a copy of the input DataFrame.

    """

    if len(data) < n:

        return data.copy()

    return data.head(n)

