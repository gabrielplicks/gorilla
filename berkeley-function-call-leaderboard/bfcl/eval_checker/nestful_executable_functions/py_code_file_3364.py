def concatenate_names(var1: str, var2: str) -> str:

    """

    Concatenates two string variables and returns the newly concatenated string.

    However, if the two string variables have the same name, then instead of

    concatenating them, return the concatenated name of the two variables.

    Args:

        var1: The first string variable.

        var2: The second string variable.

    """

    if var1 == var2:

        return var1 + var1

    else:

        return var1 + var2

