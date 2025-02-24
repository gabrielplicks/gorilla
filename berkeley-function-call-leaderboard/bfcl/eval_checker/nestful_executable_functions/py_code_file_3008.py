from typing import Callable



def arith_op(a: int, b: int, op: int) -> float:

    """

    Performs an arithmetic operation on two integers based on the value of `op`.



    Args:

        a: The first integer.

        b: The second integer.

        op: The arithmetic operation to perform.

    """

    ops: dict[int, Callable[[int, int], float]] = {

        0: lambda x, y: x + y,

        1: lambda x, y: x - y,

        2: lambda x, y: x * y,

        3: lambda x, y: x / y

    }

    return ops[op](a, b)

