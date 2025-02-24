import re

from typing import Tuple



def is_not_comment_character_threshold_met(path: str, threshold: int) -> bool:

    """Determines if the number of characters that are not comments in a Python file is greater than or equal to a threshold.



    Args:

        path: The path to the Python file.

        threshold: The minimum number of characters that are not comments.



    Returns:

        True if the number of characters that are not comments is greater than or equal to the threshold, False otherwise.

    """

    not_comment_chars_count = 0

    with open(path, 'r') as file:

        for line in file:

            if not re.match(r'^.*#.*$', line, re.IGNORECASE):

                not_comment_chars_count += len(line)



    return not_comment_chars_count >= threshold

