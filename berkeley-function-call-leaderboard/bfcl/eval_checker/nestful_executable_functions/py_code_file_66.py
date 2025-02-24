import pandas as pd



def combine_df_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:

    """Combines the columns `c2` of two dataframes `df1` and `df2` into a new column `c3`.



    Args:

        df1: The first dataframe with columns `c1` and `c2`.

        df2: The second dataframe with columns `c1` and `c2`.



    Returns:

        The modified `df1` dataframe with the new `c3` column.

    """

    df1['c3'] = df1['c2'].astype(str) + df2['c2'].astype(str)

    return df1

