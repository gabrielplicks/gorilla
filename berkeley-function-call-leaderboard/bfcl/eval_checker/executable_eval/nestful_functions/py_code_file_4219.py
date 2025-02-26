from typing import Dict, List



def generate_pv_names(motor_names: List[str]) -> Dict[str, str]:

    """Generates a dictionary of PV names based on a list of motor names.

    Args:

        motor_names: A list of motor names.

    Returns:

        A dictionary with motor names as keys and PV names as values.

    """

    pv_names = {}

    for motor_name in motor_names:

        pv_name = motor_name + '_pv'

        pv_names[motor_name] = pv_name

    return pv_names

