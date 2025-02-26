from typing import List



def separated_names(names: List[str]) -> str:

    """

    Returns a string of the names separated by commas and the word "and" before the last name.

    If the list contains only one name, return it as a string.



    Args:

        names: A list of names.

    """

    if len(names) == 1:

        return names[0]

    result = names[0]

    for name in names[1:-1]:

        result += ", " + name

    result += ", and " + names[-1]



    return result

