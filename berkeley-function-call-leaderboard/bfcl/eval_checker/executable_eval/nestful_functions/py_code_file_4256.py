from typing import List, Tuple



def read_input_file(filename: str) -> List[Tuple[str, str, str]]:

    """Reads an input file, parses each line, and creates a list of tuples with the first three elements from the parsed line.



    Args:

        filename: The name of the input file.



    Returns:

        A list of tuples containing the first three elements from each line in the input file.

    """

    result = []



    with open(filename, 'r') as f:

        for line in f:

            elements = line.split()

            if len(elements) >= 3:

                result.append(tuple(elements[:3]))



    return result

