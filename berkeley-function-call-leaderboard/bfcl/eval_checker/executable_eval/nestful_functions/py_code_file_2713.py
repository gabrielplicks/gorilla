from typing import List, Union



def build_symbol_name(args: Union[str, List[str]]) -> str:

    """Builds a symbol name in the format XX_YY_ZZ.



    Args:

        args: A string or a list of strings.



    Returns:

        A string in the format XX_YY_ZZ.

    """

    if len(args) == 1:

        return f"{args[0]}_{args[0]}_{args[0]}"

    elif len(args) == 2:

        return f"{args[0]}_{args[1]}_{args[0]}"

    elif len(args) == 3:

        return f"{args[2]}_{args[0]}_{args[1]}"

    else:

        raise ValueError("Invalid number of arguments")

