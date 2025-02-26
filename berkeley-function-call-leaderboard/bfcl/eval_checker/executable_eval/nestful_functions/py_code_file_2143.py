import pandas as pd



def add_mean(df: pd.DataFrame) -> pd.DataFrame:

    """Calculates the mean of every row's `'value'` column and adds a new column `'mean'` to the DataFrame.



    Args:

        df: The input DataFrame.



    Returns:

        The updated DataFrame with the new `'mean'` column.

    """

    df['mean'] = df.apply(lambda row: row['value'].mean(), axis=1)

    return df

