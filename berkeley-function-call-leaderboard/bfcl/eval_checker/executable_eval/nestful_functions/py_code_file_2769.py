import pandas as pd



def drop_missing_columns(df: pd.DataFrame) -> pd.DataFrame:

    """Returns a new DataFrame with only the columns that don't have any missing values.



    Args:

        df: The input DataFrame.



    Returns:

        A new DataFrame with only the columns that don't have any missing values.

    """

    return df.dropna(axis=1, how='any')

