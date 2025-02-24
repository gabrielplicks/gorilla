import math



def create_e_function() -> callable:

    """

    Creates a function that calculates the constant e for a given x value.

    Returns:

        A function that calculates the constant e for a given x value.

    """

    def e_function(x: float) -> float:

        """

        Calculates the constant e for a given x value.

        Args:

            x: A positive integer.

        """

        return math.exp(x * math.sin(x))



    return e_function

