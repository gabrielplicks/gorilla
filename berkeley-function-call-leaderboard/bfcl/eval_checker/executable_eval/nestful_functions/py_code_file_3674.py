from typing import List



def delete_value(lst: List[int], val: int) -> List[int]:

    """Deletes all occurrences of a value from a list.



    Args:

        lst: The list from which to delete the value.

        val: The value to delete from the list.



    Raises:

        ValueError: If the value is not found in the list.

    """

    try:

        while val in lst:

            lst.remove(val)

        return lst

    except ValueError:

        raise ValueError(f'{val} not found in list')

