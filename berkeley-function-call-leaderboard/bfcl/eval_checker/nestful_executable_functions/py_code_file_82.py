import re



def fix_tex_maths(text: str) -> str:

    """

    Fixes TeX maths expressions in a given string by removing leading and trailing spaces.



    Args:

        text: The input string containing TeX maths expressions.



    Returns:

        The input string with all TeX maths expressions fixed.

    """

    math_pattern = re.compile(r'\$(.*?)\$')

    def fix_math(match):

        inner_math = match.group(1).strip()

        return f'${inner_math}$'

    return math_pattern.sub(fix_math, text)

