import json

from typing import List, Dict



def load_data(filename: str) -> List[Dict[str, float]]:

    """Loads data from a file and returns it as a list of dictionaries.



    Args:

        filename: The name of the file to load data from.



    Returns:

        A list of dictionaries, where each dictionary represents a record.

    """

    with open(filename, 'r') as f:

        data = json.load(f)

    return data['records']



def save_data(records: List[Dict[str, float]], filename: str):

    """Saves a list of dictionaries to a file.



    Args:

        records: A list of dictionaries, where each dictionary represents a record.

        filename: The name of the file to save data to.

    """

    with open(filename, 'w') as f:

        json.dump({'records': records}, f, indent=2)

