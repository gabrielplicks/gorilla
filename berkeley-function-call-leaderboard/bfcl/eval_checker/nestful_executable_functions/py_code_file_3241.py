from typing import List



def split_sections(text: str, delimiter: str) -> List[str]:

    """Splits a string into sections based on the given delimiter.

    Args:

        text: The string to split.

        delimiter: The separator that defines the sections.

    """

    return text.split(delimiter)

