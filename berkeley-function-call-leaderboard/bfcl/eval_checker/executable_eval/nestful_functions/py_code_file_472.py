from typing import List, Dict



def read_and_parse(file_path: str) -> List[Dict[str, str]]:

    """Reads input data from a file and returns a list of dictionaries.



    Args:

        file_path: The path to the input file.



    Returns:

        A list of dictionaries, where each dictionary contains a single key-value pair.

    """

    data = []

    with open(file_path, "r") as file:

        for line in file:

            words = line.strip().split()

            data.append({words[0]: words[1]})

    return data

