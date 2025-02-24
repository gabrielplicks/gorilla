from typing import Union



def disambiguate_name(name: str) -> str:

    """Disambiguates a name with optional middle initial and last initial, and returns a single string that replaces the middle initial with a period.



    Args:

        name: The name with optional middle initial and last initial.



    Returns:

        A single string that replaces the middle initial with a period.

    """

    name_parts = name.split()

    if len(name_parts) > 2:

        return " ".join([name_parts[0], ".", name_parts[-1]])

    else:

        return name

