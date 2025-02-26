from typing import List, Tuple



def get_top(arr: List[Tuple[str, int]], n: int) -> dict:

    """Creates a dictionary with the top n largest amounts from a list of tuples.



    The dictionary is keyed by the unique id of the element, and the value is the amount.

    The dictionary is sorted in descending order.



    Args:

        arr: A list of tuples of the form (id, amount) with id being a unique identifier and amount being a positive integer.

        n: The number of top elements to return.

    """

    sorted_arr = sorted(arr, key=lambda x: x[1], reverse=True)  # Sort the array by amount in descending order

    top_n = sorted_arr[:n]  # Take the first n elements

    top_dict = {x[0]: x[1] for x in top_n}  # Create the dictionary using ids as keys and amounts as values

    return top_dict

