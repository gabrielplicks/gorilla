from typing import Union



def ratio_to_percentage(ratio: Union[int, float]) -> str:

    """Converts a ratio value between 0 and 1 into its equivalent percentage in a 2-decimal fixed-point format.



    Args:

        ratio: The ratio value to convert.



    Returns:

        The percentage value as a string.

    """

    percentage = round(ratio * 100, 2)

    return f'{percentage:.2f}%'

