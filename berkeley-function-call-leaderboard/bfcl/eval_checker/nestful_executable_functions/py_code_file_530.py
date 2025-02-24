from typing import List



def replace_first(lst: List[Any], item: Any, new_value: Any) -> None:

    """Replaces the first occurrence of an item in a list with a new value.

    If the item is not present in the list, adds it to the end of the list.

    If an exception occurs during execution, catches it and prints the error message.

    Args:

        lst: The list to modify.

        item: The item to replace.

        new_value: The new value to replace the item with.

    """

    try:

        index = lst.index(item)

        lst[index] = new_value

    except ValueError:

        lst.append(new_value)

