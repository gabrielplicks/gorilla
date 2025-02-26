import pandas as pd

import typing



def extract_column_from_dataframe(df: pd.DataFrame, column_name: str) -> typing.List:

    """Extracts a column from a pandas DataFrame.



    Args:

        df: The data frame to extract the column from.

        column_name: The name of the column to extract.

    """

    return df.loc[:, column_name].tolist()

