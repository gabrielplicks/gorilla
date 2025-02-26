from typing import Union



def nusselt_criterion(flow_type: str, reynolds_number: float) -> float:

    """Calculates the Nusselt criterion for the given flow type and Reynolds number.



    Args:

        flow_type: Either "laminar" or "turbulent".

        reynolds_number: The Reynolds number.



    Raises:

        Exception: If the flow type is not valid.

    """

    if flow_type == "laminar":

        return 0.023 * (reynolds_number ** 0.8)

    elif flow_type == "turbulent":

        return 0.037 * (reynolds_number ** 0.8)

    else:

        raise Exception("Invalid flow type")

