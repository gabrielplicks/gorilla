from typing import List



def name_list(names: List[str]) -> str:

    """Formats a list of names into a string of names separated by commas.

    The last two names are separated by an ampersand.

    Args:

        names: A list of names.

    """

    output = ""

    for i, name in enumerate(names):

        if i == len(names) - 2:

            output += f"{name} & "

        else:

            output += f"{name}, "

    return output.strip(", ")

