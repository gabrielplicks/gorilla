import re

from typing import Dict, Any



def macro_sub(s: str, macros: Dict[str, Any]) -> str:

    """

    Replaces all occurrences of {macro} in the string with the specified macro value.

    If a macro name is not found in the list, leave it as {macro} in the result.



    Args:

        s: The input string.

        macros: A dictionary of macro name, value pairs.

    """

    macros = dict(macros)

    return re.sub(r"{(\w+?)}", lambda m: macros.get(m.group(1), m.group()), s)

