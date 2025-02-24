from typing import Union



def convert_llvm_ir_comparison_to_short_string(operation: str, value_1: Union[str, float], value_2: Union[str, float]) -> Union[str, float]:

    """Converts an LLVM IR comparison operation into a short string that represents the condition.



    Args:

        operation: The comparison operation.

        value_1: The first value to compare.

        value_2: The second value to compare.



    Returns:

        The short string representing the condition.

    """

    if operation == 'fcmp' and value_1 == 'NaN':

        return 'nan'

    else:

        return operation

