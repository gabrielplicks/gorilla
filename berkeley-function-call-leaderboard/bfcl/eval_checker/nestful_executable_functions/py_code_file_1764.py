import time



def time_call(f: callable, n: int) -> float:

    """Calls a function `f` n times and returns the average time spent on each call.



    Args:

        f: The function to be called.

        n: The number of times to call the function.

    """

    total_time = 0

    for _ in range(n):

        start = time.perf_counter_ns()

        f()

        total_time += time.perf_counter_ns() - start

    return total_time / n

