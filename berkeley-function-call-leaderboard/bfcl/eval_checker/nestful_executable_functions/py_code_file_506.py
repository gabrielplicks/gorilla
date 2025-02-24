from typing import List

import pandas as pd



def filter_rows_by_category(df: pd.DataFrame, categories: List[str] = []) -> pd.DataFrame:

    """Filters rows in a DataFrame based on a list of categories.



    If `categories` is empty, then the function returns the original DataFrame.

    If `categories` is not empty, then the function returns only rows that match any of the supplied categories.



    Args:

        df: The input DataFrame.

        categories: A list of categories for which to filter results.

    """

    if not categories:

        return df

    return df[df['category'].isin(categories)]

