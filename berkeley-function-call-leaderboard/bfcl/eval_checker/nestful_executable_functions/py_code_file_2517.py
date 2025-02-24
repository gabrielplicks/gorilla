from typing import Union



def if_else_str_func(a: Union[str, str], b: Union[str, str], c: Union[str, str]) -> Union[str, str]:

    """Returns the string `b` if either `a` or `c` is the same as `"hello"` or `"world"`, otherwise it returns `a`.



    Args:

        a: The first string input.

        b: The second string input.

        c: The third string input.

    """

    if (a == "hello" or a == "world") or (c == "hello" or c == "world"):

        return b

    else:

        return a

