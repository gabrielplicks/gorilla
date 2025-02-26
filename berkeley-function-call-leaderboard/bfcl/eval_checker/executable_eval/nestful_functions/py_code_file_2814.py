import textwrap



def fold_string(string: str, line_limit: int) -> list[str]:

    """Folds a string into several lines based on certain conditions.

    The function breaks the string into lines based on the white space characters,

    but only if the line reaches a certain length.

    The input string could be either a single line or multiple lines.

    The output is a list of lines with the length of no more than a certain limit.



    Args:

        string: The input string to be folded.

        line_limit: The maximum length of each line.



    Returns:

        A list of lines with the length of no more than the line limit.

    """

    return textwrap.wrap(string, width=line_limit, break_long_words=False, replace_whitespace=False, drop_whitespace=True)

