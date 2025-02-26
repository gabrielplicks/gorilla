from typing import Union



def find_max_in_first_column(file_path: str) -> Union[float, None]:

    """Finds the maximum value from the first column in a file.

    Args:

        file_path: The path to the file.

    """

    max_value = None

    with open(file_path, 'r') as f:

        lines = f.readlines()

        first_column_values = [float(line.split('\t')[0]) for line in lines]

        max_value = max(first_column_values)

    return max_value

