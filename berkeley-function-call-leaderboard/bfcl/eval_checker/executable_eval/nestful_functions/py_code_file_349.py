import ast



def string_format_args(args: list[str]) -> str:

    """Formats a string using the given keyword arguments.



    Args:

        args: A list of strings to be parsed as keyword arguments.

    """

    parsed_args = [ast.literal_eval(arg) for arg in args]

    formatted_args = ", ".join([f"{key}={value}" for key, value in parsed_args])



    return formatted_args

