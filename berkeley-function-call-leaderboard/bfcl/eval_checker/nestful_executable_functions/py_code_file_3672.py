from typing import Any



class MyList(list):

    def __len__(self):

        return super().__len__()



def len_customized(c: Any) -> int:

    """Calculates the length of a collection.



    Args:

        c: The collection to calculate the length of.



    Returns:

        The length of the collection.

    """

    return c.__len__()



custom_list = MyList([1, 2, 3, 4, 5])

print(len_customized(custom_list))  # Output: 5

