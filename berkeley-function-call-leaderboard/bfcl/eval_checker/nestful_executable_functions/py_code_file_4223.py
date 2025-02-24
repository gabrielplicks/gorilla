import re

from typing import List, Tuple



def parse_string_to_tuples(input_string: str) -> List[Tuple[int, int]]:

    """Parses a string of comma-separated numbers and returns a list of tuples with the numbers and their lengths.



    Args:

        input_string: A string of comma-separated numbers.



    Returns:

        A list of tuples where the first element in each tuple is the number and the second element is the length of the number.

    """

    pattern = r"(\d+)"  # Regular expression to match numbers

    matches = re.findall(pattern, input_string)  # Extract numbers from the input string

    return [(int(number), len(number)) for number in matches]  # Create tuples with number and length

