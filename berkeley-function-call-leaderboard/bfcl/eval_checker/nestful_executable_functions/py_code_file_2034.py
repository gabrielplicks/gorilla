import numpy as np



def proportional_controller(kp: float, omega: float, zeta: float) -> float:

    """Calculates the magnitude of the transfer function of a proportional-only controller.



    Args:

        kp: The proportional gain constant.

        omega: The frequency where the controller is evaluated.

        zeta: The damping ratio, a non-negative real number.



    Returns:

        The magnitude of the transfer function at the given frequency and damping, which is proportional to the controller gain.

    """

    transfer_function = kp

    transfer_function_mag = np.abs(transfer_function)



    return transfer_function_mag

