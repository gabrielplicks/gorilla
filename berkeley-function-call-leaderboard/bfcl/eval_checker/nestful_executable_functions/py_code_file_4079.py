from typing import Union



def get_digit_sum(num: Union[int, float]) -> Union[int, str]:

    """Computes the sum of all digits in a given integer number.



    Args:

        num: The input number.



    Returns:

        The sum of all digits in the input number. If the input number is 0,

        it returns a string "Input is 0, no digit to sum up".

    """

    if num == 0:

        return "Input is 0, no digit to sum up"



    is_negative = num < 0

    num = abs(num)



    digit_sum = 0

    while num > 0:

        digit = num % 10

        num //= 10

        digit_sum += digit



    if is_negative:

        digit_sum *= -1



    return digit_sum

