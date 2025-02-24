import csv

from typing import List



def parse_csv_first_row(csv_string: str) -> List[str]:

    """Parses a string containing a CSV file's first row and returns a list of column names.



    The function removes surrounding quotes and unescapes any escaped quotes in the column names.



    Args:

        csv_string: The string containing the CSV file's first row.



    Returns:

        A list of column names.

    """

    reader = csv.reader(csv_string.splitlines(), delimiter=',')

    columns = next(reader)

    return [col.strip('"').replace('""', '"') for col in columns]

