import re

import pandas as pd



def filter_dataframe_by_name(df: pd.DataFrame) -> pd.DataFrame:

    """Filters a pandas dataframe based on a regular expression match in the `name` column.



    The regular expression accepts at least one digit and at least one alphabetic character.



    Args:

        df: The input dataframe.



    Returns:

        A new dataframe containing only rows whose values in the `name` column match the regular expression.

    """

    regex = re.compile(r"(?=.*\d)(?=.*[a-zA-Z]).*")

    filtered_df = df[df["name"].str.match(regex)]

    return filtered_df

