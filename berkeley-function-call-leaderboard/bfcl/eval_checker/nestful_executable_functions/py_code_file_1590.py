from typing import Optional



def get_output(input: int) -> Optional[int]:

    """

    Returns the output corresponding to the given input based on the input-output pairs.

    If the input is not present in the input-output pairs, returns None.



    Args:

        input: The input for which the output is to be determined.



    Returns:

        The output corresponding to the input, or None if the input is not present in the input-output pairs.

    """

    if input == 2:

        return 2**2

    elif input == 5:

        return 5**2

    else:

        return None

