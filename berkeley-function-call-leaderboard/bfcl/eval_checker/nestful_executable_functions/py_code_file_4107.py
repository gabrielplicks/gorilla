import csv

from typing import List, Dict



def convert_and_sort_csv_data(csv_string: str, sort_column: str) -> List[Dict[str, str]]:

    """Converts a CSV string to a list of objects and sorts them based on a specific column value.

    Args:

        csv_string: The CSV string to convert and sort.

        sort_column: The column to sort the objects by.

    """

    reader = csv.DictReader(csv_string.splitlines())

    sorted_data = sorted(list(reader), key=lambda row: row[sort_column])

    return sorted_data

