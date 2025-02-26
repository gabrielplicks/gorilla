from typing import List

from collections import deque



def translate(queue: deque) -> str:

    """Translates a queue of numbers into a string by concatenating each number's digits.

    If the number is negative, adds a '-' before its digits. The concatenated string

    should be empty when the queue is empty.

    Args:

        queue: A deque containing the numbers to be translated.

    Returns:

        A string containing the concatenated digits of the numbers in the queue.

    """

    result = ""

    while queue:

        num = queue.popleft()

        if num < 0:

            result += "-"

        result += str(abs(num))

    return result

