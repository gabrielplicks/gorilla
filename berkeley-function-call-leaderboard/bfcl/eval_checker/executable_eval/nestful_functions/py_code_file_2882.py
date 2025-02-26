from typing import List



def csv_format(string_list: List[str]) -> str:

    """Converts a list of strings into a string of comma-separated values (CSV) format.

    If a string includes a comma, double quotation marks, or a line break, wrap the string in double quotation marks,

    with any internal double quotation marks escaped with another double quotation mark.

    Args:

        string_list: A list of strings to be converted to CSV format.

    """

    csv_list = []

    for string in string_list:

        if ',' in string or '"' in string or '\n' in string:

            string = string.replace('"', '""')

            string = f'"{string}"'

        csv_list.append(string)

    return ','.join(csv_list)

