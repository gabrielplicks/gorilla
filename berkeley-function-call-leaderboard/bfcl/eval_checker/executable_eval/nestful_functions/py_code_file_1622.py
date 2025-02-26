from typing import Dict

import pandas as pd



def split_by_year(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:

    """Splits a data frame into a dictionary of data frames based on unique years.



    Args:

        df: A data frame with columns 'year', 'month', and 'value'.



    Returns:

        A dictionary where the keys are the unique years from the data frame and the values are data frames containing all rows with the corresponding year.

    """

    years = df['year'].unique()

    d = {}

    for year in years:

        d[year] = df.loc[df['year'] == year, ['month', 'value']]

    return d

