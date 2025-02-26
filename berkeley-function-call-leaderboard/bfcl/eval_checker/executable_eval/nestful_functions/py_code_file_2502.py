import re



def remove_linebreaks(text: str) -> str:

    """

    Removes linebreaks from a given string except for the last line.



    Args:

        text: The input string.

    """

    return re.sub(r"(\r?\n)(?!.+\r?\n)", "", text)

