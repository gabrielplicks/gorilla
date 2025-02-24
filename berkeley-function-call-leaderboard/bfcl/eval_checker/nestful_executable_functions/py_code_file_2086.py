import pandas as pd

import re



def find_columns_with_string(df: pd.DataFrame, string: str) -> list:

    """

    Finds column names in a DataFrame that contain a specified string.



    Args:

        df: The DataFrame to search.

        string: The string to search for in the column names.

    """

    column_names = []

    for column_name in df.columns:

        if re.search(string, column_name):

            column_names.append(column_name)

    return column_names

