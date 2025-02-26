import pandas as pd



def is_dataframe_empty(df: pd.DataFrame) -> bool:

    """Checks whether a pandas DataFrame is empty or not.



    Args:

        df: The DataFrame to check.



    Returns:

        True if the DataFrame is empty and False otherwise.

    """

    return len(df) == 0

