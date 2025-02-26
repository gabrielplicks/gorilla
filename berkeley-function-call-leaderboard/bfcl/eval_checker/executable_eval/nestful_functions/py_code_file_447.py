import pandas as pd



def get_new_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    """Returns a new DataFrame with two columns, `x_len` and `y_positive`, where `x_len` is the length of each string in `x` and `y_positive` is a Boolean column indicating whether the corresponding value in `y` is positive.



    Args:

        df: The input DataFrame with columns `x` and `y`.

    """

    new_df = pd.DataFrame()

    new_df['x_len'] = df['x'].apply(len)

    new_df['y_positive'] = df['y'].apply(lambda x: x > 0)

    return new_df

