import pandas as pd

import re



def remove_columns_by_pattern(df: pd.DataFrame, pattern: str) -> pd.DataFrame:

    """Removes columns from a given pandas data frame that match a given pattern.

    Args:

        df: The input data frame.

        pattern: The regular expression pattern to match.

    """

    columns_to_drop = df.columns[df.columns.str.match(pattern)]

    df = df.drop(columns=columns_to_drop)



    return df

