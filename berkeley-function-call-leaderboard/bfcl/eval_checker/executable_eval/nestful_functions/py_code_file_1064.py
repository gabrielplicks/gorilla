import re

from typing import List



def clean_name(name: str) -> str:

    """Cleans a name by removing leading and trailing spaces, replacing multiple spaces with a single space, and converting all characters to lowercase.



    Args:

        name: The name to be cleaned.



    Returns:

        The cleaned name.

    """

    name = name.strip()

    name = re.sub(r'\s+', ' ', name)

    name = name.lower()

    return name



def clean_names(names: List[str]) -> List[str]:

    """Cleans a list of names by removing leading and trailing spaces, replacing multiple spaces with a single space, and converting all characters to lowercase.



    Args:

        names: The list of names to be cleaned.



    Returns:

        The list of cleaned names.

    """

    cleaned_names = [clean_name(name) for name in names]

    return cleaned_names

