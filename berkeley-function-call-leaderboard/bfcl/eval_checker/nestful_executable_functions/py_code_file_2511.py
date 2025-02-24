from typing import List



def add_anchor_tags(input_string: str) -> str:

    """Creates a new string with anchor tags around each word in the input string.

    Args:

        input_string: The input string to be processed.

    Returns:

        A new string with anchor tags around each word.

    """

    words: List[str] = input_string.split()

    anchor_tags: List[str] = ['<a href="#{}">{}</a>'.format(word, word) for word in words]

    output_string: str = ' '.join(anchor_tags)

    return output_string

