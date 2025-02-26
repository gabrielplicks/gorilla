from typing import List



def append_length(input_string: str) -> str:

    """Appends the length of each string in a comma-separated list to the end of the string.



    Args:

        input_string: A comma-separated list of strings.



    Returns:

        A new comma-separated list of strings with the length of each string appended to the end of the string.

    """

    output_list: List[str] = []

    input_list: List[str] = input_string.split(',')

    for string in input_list:

        output_list.append(f'{string}{len(string)}')

    return ','.join(output_list)

