from typing import Union



def classify(x: Union[int, float]) -> str:

    """Classifies a floating-point number `x` and returns a string.



    Args:

        x: The floating-point number to classify.



    Returns:

        A string representing the classification of `x`.

    """

    if x <= 0:

        return f"bad: {x:.6f}"

    elif 0 < x < 0.5:

        return f"marginal: {x:.6f}"

    elif 0.5 <= x < 0.7:

        return f"good: {x:.6f}"

    else:

        return f"excellent: {x:.6f}"

