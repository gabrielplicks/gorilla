from typing import List



def format_sequence_as_rst(sequence: List[str]) -> str:

    """Converts a list of strings representing a sequence of numbers into a list of strings representing the sequence in reStructuredText format.

    Args:

        sequence: A list of strings representing a sequence of numbers.

    """

    output = ''

    for index, string in enumerate(sequence, 1):

        output += f'{index}. {string}\n'

    return output

