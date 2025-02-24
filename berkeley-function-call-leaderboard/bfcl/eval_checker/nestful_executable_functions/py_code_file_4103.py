from typing import Dict



def extract_step_components(input_string: str) -> Dict[str, str]:

    """Extracts the components of a step name from a string.



    The input string should be in the format "user.stepname.parameter".

    The function returns a dictionary with the keys "user", "step", and

    "parameter" and the corresponding values.



    Args:

        input_string: The input string in the format "user.stepname.parameter".



    Returns:

        A dictionary with the keys "user", "step", and "parameter" and the

        corresponding values.

    """

    user, step, parameter = input_string.split('.')

    return {'user': user, 'step': step, 'parameter': parameter}

