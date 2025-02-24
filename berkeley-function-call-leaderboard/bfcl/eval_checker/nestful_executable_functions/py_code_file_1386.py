from typing import List



def split_string_by_underscore(input_string: str) -> List[str]:

    """Splits a string by underscores and returns a list of strings.



    If the input string does not contain underscores, the function returns a

    single-element list containing the input string.



    Args:

        input_string: The string to split.



    Returns:

        A list of strings.

    """

    split_string = input_string.split('_')



    if len(split_string) == 1:

        return [input_string]



    return split_string

