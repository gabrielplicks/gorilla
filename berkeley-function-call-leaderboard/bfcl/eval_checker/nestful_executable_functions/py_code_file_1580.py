from typing import List



def ode_model(r: float, C0: float, t: List[float]) -> List[float]:

    """Calculates the dependent variable C at different times using Euler's method.



    Args:

        r: The parameter r in the ODE model.

        C0: The initial value of the dependent variable C.

        t: A list of time values.



    Returns:

        A list of the dependent variable C at different times.

    """

    C = [C0]

    for i in range(1, len(t)):

        C_next = C[i - 1] + r * C[i - 1] * (t[i] - t[i - 1])

        C.append(C_next)



    return C

