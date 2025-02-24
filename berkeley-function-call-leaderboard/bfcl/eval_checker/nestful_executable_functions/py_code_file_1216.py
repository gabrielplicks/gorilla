import pandas as pd



def remove_bad_column(df: pd.DataFrame) -> pd.DataFrame:

    """Removes the 'badColumn' column if it exists in the given DataFrame and returns a new DataFrame without 'badColumn'.



    Args:

        df: The input DataFrame.



    Returns:

        A new DataFrame without the 'badColumn' column.

    """

    if 'badColumn' in df.columns:

        df = df.drop(columns=['badColumn'])

    return df

