from typing import Dict

import pandas as pd



def map_category_to_book(df: pd.DataFrame, lookup: Dict[int, str]) -> pd.DataFrame:

    """Maps the category of each book in a dataframe to its corresponding category name.



    Args:

        df: A pandas dataframe with `category_id` and `category_name` columns.

        lookup: A dictionary mapping category IDs to category names.



    """

    df['category'] = df['category_id'].apply(lambda x: lookup[x])

    return df

