from typing import List



def collect_strings(strings: List[str]) -> str:

    """Concatenates a list of strings into a single string, separating each element with a comma followed by a space, except for the last element, which should be separated by "and".



    Args:

        strings: A list of strings to be concatenated.



    Returns:

        A single concatenated string.

    """

    if len(strings) == 1:

        return strings[0]



    return ', '.join(strings[:-1]) + ' and ' + strings[-1]

