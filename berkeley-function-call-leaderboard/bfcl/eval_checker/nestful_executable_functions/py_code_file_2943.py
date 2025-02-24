from typing import List, Tuple, Union



def fill_missing(iterable: Union[List, Tuple], default: Union[int, float, str]) -> List:

    """Fills missing elements in an iterable with a default value.



    Args:

        iterable: The iterable to fill missing elements in.

        default: The default value to use for missing elements.



    Raises:

        TypeError: If the iterable is not a list or a tuple.

    """

    if not isinstance(iterable, (list, tuple)):

        raise TypeError('Iterable must be a list or a tuple')



    filled_list = []

    for element in iterable:

        if element is None:

            filled_list.append(default)

        else:

            filled_list.append(element)

    return filled_list

