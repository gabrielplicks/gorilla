from typing import AnyStr



def generate_c_for_loop_code(format_str: AnyStr) -> str:

    """Generates the C code for a for loop that prints the numbers from 1 to 10.



    Args:

        format_str: The format string used to format the numbers as strings before printing.

    """

    code = []

    code.append("for (int i = 1; i <= 10; i++) {")

    code.append("    printf(\"{}\", i);".format(format_str))

    code.append("}")

    return "\n".join(code)

