import inspect



def check_functions(func1: callable, func2: callable) -> bool:

    """Checks if the names of two functions are the same.



    Args:

        func1: The first function to check.

        func2: The second function to check.

    """

    func1_args = inspect.getfullargspec(func1)

    func2_args = inspect.getfullargspec(func2)

    if func1.__name__ == func2.__name__:

        return True

    if len(func1_args.args) == len(func2_args.args):

        return True

    if func1_args.defaults is not None and func2_args.defaults is not None:

        if len(func1_args.defaults) == len(func2_args.defaults):

            return True

    return False

