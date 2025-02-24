from typing import List, Tuple, Union



def repeat_elements(my_list: Union[List, Tuple], n: int) -> Union[List, Tuple]:

    """Repeats each element in a list or tuple `n` times.



    Args:

        my_list: The list or tuple to repeat.

        n: The number of times to repeat each element.



    Returns:

        A list or tuple containing `n` copies of each element in `my_list`.

    """

    if n < 1:

        return []

    if len(my_list) == 0:

        return []

    output = []

    for element in my_list:

        for _ in range(n):

            output.append(element)

    return output

