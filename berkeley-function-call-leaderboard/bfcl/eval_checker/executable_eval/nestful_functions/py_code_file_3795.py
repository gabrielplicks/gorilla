from typing import List



def create_id_list(my_list: List[str]) -> List[str]:

    """Creates a new list `my_id_list` that contains ids of the form `my_id_<element>`.



    Args:

        my_list: The list of elements.



    Returns:

        A list of ids.

    """

    my_id_list = []



    for element in my_list:

        my_id = f"my_id_{element}"

        my_id_list.append(my_id)



    return my_id_list

