import pandas as pd

import re



def filter_df_by_regex(df: pd.DataFrame, regex: str, columns: list) -> pd.DataFrame:

    """Filters a given pandas DataFrame based on a regex matching pattern.



    Args:

        df: A pandas DataFrame to be filtered.

        regex: A regular expression pattern used for matching.

        columns: A list of columns to match the pattern in.



    Returns:

        A new DataFrame that contains only the rows that match the regular expression pattern in the specified columns.

    """

    return df.filter(regex=regex, axis=1)

