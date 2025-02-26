from datetime import datetime

from typing import List, Dict



def convert_to_iso8601(input_dict: List[Dict[str, str]]) -> List[Dict[str, str]]:

    """Converts the date/time key value to ISO8601 format and adds a new key named 'created_at'.



    Args:

        input_dict: A list of dictionaries containing a 'date' key.



    Returns:

        A list of dictionaries with the 'created_at' key added.

    """

    output_dict = []

    for item in input_dict:

        date = datetime.strptime(item['date'], '%Y-%m-%d')

        iso8601_date = date.isoformat()

        item['created_at'] = iso8601_date

        output_dict.append(item)

    return output_dict

