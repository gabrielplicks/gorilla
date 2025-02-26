import pandas as pd

from typing import Dict, Union



def get_highest_rank(dataframe: pd.DataFrame) -> Union[pd.Series, None]:

    """Returns the object with the highest rank from a Pandas dataframe.



    Args:

        dataframe: The dataframe containing the objects.



    Returns:

        The object with the highest rank, or None if no object has a rank.

    """

    highest_rank = -1

    highest_rank_object = None

    for _, row in dataframe.iterrows():

        if 'rank' in row:

            rank = row['rank']

        elif 'rank_data' in row:

            rank = row['rank_data']['value'].min()

        else:

            continue

        if rank > highest_rank:

            highest_rank = rank

            highest_rank_object = row

    return highest_rank_object

