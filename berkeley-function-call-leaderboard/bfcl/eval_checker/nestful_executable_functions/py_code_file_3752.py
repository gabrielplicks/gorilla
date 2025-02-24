import re

import typing



def extract_header(csv_file: str) -> typing.List[str]:

    """Extracts the first line of a csv file that starts with the character 'c' (case-insensitive) and contains at least one comma.



    Args:

        csv_file: The path to the csv file.



    Returns:

        A list of column names if a header line is found, otherwise an empty list.

    """

    with open(csv_file, 'r') as file:

        content = file.read()

        header_pattern = re.compile(r'^c[^,]*,.*$', re.IGNORECASE)

        match = re.search(header_pattern, content)

        if match:

            header_line = match.group()

            header_columns = header_line.split(',')

            return header_columns

    return []

