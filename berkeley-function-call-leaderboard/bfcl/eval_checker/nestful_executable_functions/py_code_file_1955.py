def draw_staircase(n: int) -> None:

    """Draws a staircase with n rows and n characters in each row.



    Args:

        n: The number of rows in the staircase.

    """

    for i in range(n, 0, -1):

        print(' ' * i + '#' * (n - i + 1))

