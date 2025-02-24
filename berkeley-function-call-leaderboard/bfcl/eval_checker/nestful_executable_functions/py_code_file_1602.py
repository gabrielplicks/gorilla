from typing import Union



def process_strings(input_data: Union[str, int, float]) -> str:

    """Processes a number of strings with `str.strip()` and concatenates the results.

    If a string is empty (i.e., `''`), it is ignored. If the input is an integer or a float,

    it is converted to a string first.



    Args:

        input_data: The input data to be processed.



    Returns:

        The concatenated string.

    """

    return '+'.join([s.strip() for s in str(input_data).split('+') if s.strip()])

