from typing import List, Union



def square_values(values: List[Union[int, str]]) -> List[int]:

    """

    Takes a list of numbers or strings as input and returns a list of the squared values.

    Args:

        values: A list of numbers or strings.

    """

    return [int(value)**2 if isinstance(value, str) else value**2 for value in values]

