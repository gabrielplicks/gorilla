import pandas as pd



def remove_columns_with_unique_values(df: pd.DataFrame) -> pd.DataFrame:

    """Removes columns from a DataFrame that contain only unique values.



    Args:

        df: The input DataFrame.



    Returns:

        The same DataFrame with selected columns removed.

    """

    for col in df.columns:

        if df[col].nunique() == df.shape[0]:

            df.drop(col, axis=1, inplace=True)

    return df

