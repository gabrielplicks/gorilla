import csv

from typing import List, Tuple



def read_names(csv_file: str) -> List[Tuple[str, str]]:

    """

    Read a CSV file and return a list of tuples of strings, where each tuple contains the first and last names of a person.



    Args:

        csv_file: Path to the CSV file.



    Returns:

        List of tuples of strings, where each tuple contains the first and last names of a person.

    """

    with open(csv_file, 'r') as f:

        reader = csv.reader(f)

        next(reader)  # Skip the header row

        names = []

        for row in reader:

            first_name, last_name = row[0], row[1]

            names.append((first_name, last_name))

    return names

