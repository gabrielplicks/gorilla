import re

import typing



def do_something_special(s: str) -> str:

    """

    Modifies a string based on its content.

    If the string is 'hello world' with optional leading or trailing whitespace,

    it returns 'hi world'. If the string is 'hello there' with optional leading

    or trailing whitespace, it returns 'hi there'. Otherwise, it returns

    'nothing special'.



    Args:

        s: The input string.

    """

    if re.match(r'^\s*hello world\s*$', s):

        return 'hi world'

    elif re.match(r'^\s*hello there\s*$', s):

        return 'hi there'

    else:

        return 'nothing special'

