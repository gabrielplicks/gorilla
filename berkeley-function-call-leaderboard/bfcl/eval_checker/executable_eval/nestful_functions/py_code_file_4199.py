from typing import Dict



def read_file_into_dict(file_path: str) -> Dict[str, list]:

    """

    Reads the data from a file with the format 'key -> value1, value2, value3'

    into a dictionary with the data structure { key1: [list of values], key2: [list of values], ... }

    Args:

        file_path: The path to the file to read.

    """

    with open(file_path, 'r') as file:

        lines = file.readlines()

        data = {}

        for line in lines:

            key, values = line.strip().split(' -> ')

            data.update({key: values.split(', ')})

    return data

