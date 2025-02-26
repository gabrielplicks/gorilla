import math



def roots(a: float, b: float, c: float) -> tuple:

    """Computes the roots of a quadratic equation ax^2 + bx + c = 0.



    Args:

        a: The coefficient of the quadratic term.

        b: The coefficient of the linear term.

        c: The constant term.



    Raises:

        Exception: If the equation has no real-valued roots.

    """

    discriminant = b**2 - 4*a*c

    if discriminant < 0:

        raise Exception("The equation has no real-valued roots.")



    sqrt_discriminant = math.sqrt(discriminant)

    root_1 = (-b + sqrt_discriminant) / (2*a)

    root_2 = (-b - sqrt_discriminant) / (2*a)



    return root_1, root_2

