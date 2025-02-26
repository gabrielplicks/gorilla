import re

from typing import AnyStr



def is_yes_no_question(question: AnyStr) -> bool:

    """Checks if a given string is a yes/no question.



    Args:

        question: The string to check.



    Returns:

        True if the string is a yes/no question, False otherwise.

    """

    question = question.lower()

    question = re.sub(r'[^\w\s]', '', question)

    return question.startswith("is ") or question.startswith("are ")

