from typing import List



def greet_names(names: List[str]) -> str:

    """Greets a list of names.

    Args:

        names: A list of names.

    """

    if len(names) == 1:

        return f"Hello there {names[0]}!"

    elif len(names) == 2:

        return f"Hello there {names[0]} and {names[1]}!"

    else:

        names_str = ", ".join(names[:-1]) + " and " + names[-1]

        return f"Hello there {names_str}!"

