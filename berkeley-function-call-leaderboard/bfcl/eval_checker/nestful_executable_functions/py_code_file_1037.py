import re

from typing import List



def madlib_transformer(template: str, words: List[str]) -> str:

    """Fills in the blanks in a template string with words from a list.



    Args:

        template: The template string containing underscores (_) denoting blanks to be filled in.

        words: The list of words to fill in the blanks.



    Returns:

        The filled-in template string.

    """

    blanks = re.findall(r'_+', template)

    for word in words:

        next_blank_index = template.find('_')

        if next_blank_index == -1:

            break

        template = template.replace('_', word, 1)



    return template

