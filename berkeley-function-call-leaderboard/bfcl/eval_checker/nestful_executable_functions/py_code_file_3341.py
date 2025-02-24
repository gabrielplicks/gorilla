from typing import Union



def label_value(value: Union[int, float]) -> str:

    """Translates a numeric value into a corresponding text-based label.



    Args:

        value: The numeric value to be translated.



    Returns:

        A text-based label corresponding to the input value.

    """

    if value < 0:

        return "negative"

    elif value == 0:

        return "zero"

    elif 0 < value < 10:

        return "positive small"

    else:

        return "positive large"

