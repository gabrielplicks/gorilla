import csv

from typing import List, Dict



def import_data(file_path: str) -> List[Dict[str, str]]:

    """Reads a CSV file and returns a list of dictionaries, where each dictionary represents a row of data.



    Args:

        file_path: The path to the CSV file.



    Returns:

        A list of dictionaries, where each dictionary represents a row of data.

        The keys of the dictionaries are the column names, and the values are the corresponding data.

    """

    with open(file_path) as f:

        reader = csv.DictReader(f, skipinitialspace=True)

        return [dict(row) for row in reader]

