import pandas as pd

from typing import Union



def rename_column(df: pd.DataFrame) -> Union[pd.DataFrame, str]:

    """Renames the column 'A' to 'X' in the given DataFrame.



    If the column 'A' does not exist, the function returns a message indicating that the column does not exist.



    Args:

        df: The DataFrame to rename the column in.



    Returns:

        The DataFrame with the renamed column or a message indicating that the column does not exist.

    """

    if 'A' in df.columns:

        df = df.rename(columns={'A': 'X'})

        return df

    else:

        return "Column 'A' does not exist."

