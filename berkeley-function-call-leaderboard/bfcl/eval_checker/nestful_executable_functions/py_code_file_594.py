import pandas as pd



def filter_and_convert(df: pd.DataFrame) -> pd.DataFrame:

    """Filters out non-numeric values in a Pandas DataFrame and converts the remaining values to floats.



    Args:

        df: The input DataFrame.



    Returns:

        A new DataFrame with the same columns, but with non-numeric values filtered out and the remaining values converted to floats.

    """

    df['x'] = df['x'].apply(lambda x: pd.to_numeric(x, errors='coerce'))

    df['y'] = df['y'].apply(lambda y: pd.to_numeric(y, errors='coerce'))

    df = df.dropna()

    return df

