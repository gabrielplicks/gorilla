from typing import List



def sanitize_python_code(code: str) -> str:

    """Sanitizes a string of Python code by removing comments and leading/trailing whitespace.



    Args:

        code: The input Python code as a string.



    Returns:

        The sanitized Python code as a string.

    """

    sanitized_code: List[str] = []

    for line in code.splitlines():

        line = line.split("#")[0].strip()

        line = line.strip()

        if line:

            sanitized_code.append(line)



    return "\n".join(sanitized_code)

