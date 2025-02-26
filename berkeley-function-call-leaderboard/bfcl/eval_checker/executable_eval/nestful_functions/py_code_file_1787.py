import operator



def operator_to_function(operator_string: str) -> operator:

    """Converts an operator string to its corresponding function.



    Args:

        operator_string: The operator string to convert.

    """

    operators = {

        '+': operator.add,

        '-': operator.sub,

        '*': operator.mul

    }

    return operators[operator_string]

