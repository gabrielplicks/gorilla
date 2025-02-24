from datetime import datetime

from typing import List, Dict



def get_index_dict(dates: List[datetime], values: List[int]) -> Dict[datetime, int]:

    """Creates a dictionary where the keys are dates and the values are the index of the associated value.



    Args:

        dates: A list of datetime objects.

        values: A list of values associated with the dates.



    Returns:

        A dictionary where the keys are dates and the values are the index of the associated value.

    """

    return {date: i for i, date in enumerate(dates)}

