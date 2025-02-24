import pandas as pd



def concat_dfs(df1: pd.DataFrame, df2: pd.DataFrame, col: str) -> pd.DataFrame:

    """Concatenates two data frames (df1 and df2) on a given column (col) only if the two data frames have the same length and the same value for the given column. If the data frames do not meet these criteria, returns an empty data frame.



    Args:

        df1: The first data frame.

        df2: The second data frame.

        col: The column to concatenate on.

    """

    if len(df1) == len(df2) and df1[col].equals(df2[col]):

        return pd.concat([df1, df2], axis=1)

    else:

        return pd.DataFrame()

