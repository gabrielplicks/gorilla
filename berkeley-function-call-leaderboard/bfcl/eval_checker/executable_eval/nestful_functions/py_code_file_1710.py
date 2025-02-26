import datetime

from typing import Dict, List, Union



def map_date(d: Dict[str, Union[int, datetime.datetime]]) -> Dict[str, Union[int, str]]:

    """Maps a dictionary with 'id' and 'date' keys to a new dictionary with 'id' and 'date_formatted' keys.

    If the original 'date' is missing, then the new 'date_formatted' is 'unknown'.

    Args:

        d: A dictionary with 'id' and 'date' keys.

    Returns:

        A dictionary with 'id' and 'date_formatted' keys.

    """

    if 'date' in d:

        date_formatted = d['date'].strftime('%Y-%m-%d')

    else:

        date_formatted = 'unknown'



    return {'id': d['id'], 'date_formatted': date_formatted}

