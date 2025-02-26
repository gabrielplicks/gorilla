from typing import List, Optional



def convert_slice_to_list(slice_obj: slice) -> List[int]:

    """Converts a slice object to a list of integers.



    Args:

        slice_obj: The slice object to convert.



    Returns:

        A list of integers.

    """

    int_list: List[int] = []

    if slice_obj.step is not None:

        step: int = slice_obj.step

    else:

        step: int = 1

    if slice_obj.stop <= slice_obj.start:

        return int_list

    for i in range(slice_obj.start, slice_obj.stop, step):

        int_list.append(i)



    return int_list

