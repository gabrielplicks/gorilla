from typing import Union



def format_plural(count: int, singular: str, plural: str) -> str:

    """Formats a string based on whether the count is one or greater than one.

    Args:

        count: The count of the object.

        singular: The singular form of the object.

        plural: The plural form of the object.

    """

    if count == 1:

        return f'{count} {singular}'

    elif count > 1:

        return f'{count} {plural}'

    else:

        return f'0 {plural}'

