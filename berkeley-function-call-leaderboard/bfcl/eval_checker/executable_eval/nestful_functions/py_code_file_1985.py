from typing import AnyStr



def insert_value(string: AnyStr, value: AnyStr) -> AnyStr:

    """Inserts a target value into a string at the location of the first occurrence of the character `"%"`.

    Args:

        string: The input string.

        value: The target value to be inserted.

    """

    return string.replace("%", value, 1)

