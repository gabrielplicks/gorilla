import pandas as pd



def get_only_name(df: pd.DataFrame) -> pd.DataFrame:

    """Returns a Pandas DataFrame consisting of the `id` and `name` columns, where the `id` column has been renamed to `user_id`.



    Args:

        df: The input Pandas DataFrame.



    Returns:

        A Pandas DataFrame with the `id` and `name` columns, where the `id` column has been renamed to `user_id`.

    """

    filtered_df = df.loc[:, ['id', 'name']]

    renamed_df = filtered_df.rename(columns={'id': 'user_id'})

    return renamed_df

