from typing import List, Union



def add_tab(text: Union[str, List[str]]) -> str:

    """Modifies the text by inserting a tab at the beginning of each line.

    Args:

        text: A string containing multiple lines of text or a list of strings.

    """

    if isinstance(text, str):

        lines = text.split("\n")

    else:

        lines = text

    lines = ["\t" + line for line in lines]

    return "\n".join(lines)

