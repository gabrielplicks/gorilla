import csv

from typing import List



def parse_csv(file_path: str) -> List[List[str]]:

    """Parses a CSV file and returns a list of lists representing the parsed contents.

    Each nested list represents a row in the CSV file, with each element in the list being a string representing a column's data element.



    Args:

        file_path: The path to the CSV file.

    """

    with open(file_path, 'r', newline='') as csv_file:

        csv_reader = csv.reader(csv_file)

        parsed_csv = [list(row) for row in csv_reader]

        return parsed_csv

