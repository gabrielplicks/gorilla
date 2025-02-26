from typing import Union



def pad_num(num: Union[int, float], pad_len: int) -> str:

    """Pads a number with leading zeros to make it have a minimum length of `pad_len`.



    Args:

        num: The number to pad.

        pad_len: The minimum length of the padded number.

    """

    num_str = str(num)

    if len(num_str) < pad_len:

        num_str = '{:0{}}'.format(num, pad_len)

    return num_str

