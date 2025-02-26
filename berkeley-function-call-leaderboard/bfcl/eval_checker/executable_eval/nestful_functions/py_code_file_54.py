def assign_color(count: int) -> str:

    """Assigns a color to the count based on the number of items in the list.



    Args:

        count: The number of items in the list.



    Returns:

        A string that describes the color of the count.

    """

    if count <= 0:

        msg = "The number of items in the list is out of range."

    elif count <= 5:

        msg = "The number of items in the list is blue."

    elif count <= 10:

        msg = "The number of items in the list is green."

    else:

        msg = "The number of items in the list is out of range."

    return msg

