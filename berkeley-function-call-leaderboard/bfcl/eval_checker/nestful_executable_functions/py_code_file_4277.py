import pandas as pd

from typing import List



def process_purchases(df: pd.DataFrame) -> pd.DataFrame:

    """Processes the purchases data and returns a new DataFrame with grouped items for each user.



    Args:

        df: The input DataFrame with columns 'user_id', 'item_id', and 'purchase_count'.



    Returns:

        A new DataFrame with columns 'user_id' and 'items' that contains the items purchased by each user.

        The 'items' column is a list of the items purchased by each user.

    """

    if df.empty:  # Check if the DataFrame is empty

        return pd.DataFrame(columns=['user_id', 'items'])

    df_out = df.groupby('user_id')['item_id'].apply(list).reset_index(name='items')



    return df_out

