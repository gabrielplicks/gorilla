import pandas as pd

from typing import Dict



def calculate_people_per_floor(df: pd.DataFrame) -> Dict[int, int]:

    """Calculates the total number of people on each floor using a dictionary.



    Args:

        df: A Pandas DataFrame with two columns: `floor` and `people`.



    Returns:

        A dictionary of the total number of people on each floor.

        The keys are the unique values in the `floor` column,

        and the values are the sum of the `people` column for each unique floor.

    """

    return df.groupby('floor')['people'].sum().to_dict()

