from typing import Union



def format_salary(currency: str, salary: Union[int, float], format_type: str) -> str:

    """Formats a salary according to the given currency and format type.



    Args:

        currency: The currency of the salary.

        salary: The salary amount.

        format_type: The format type of the salary.



    Returns:

        The formatted salary.

    """

    if not currency:

        return "Unknown currency"

    elif format_type not in ("bracket", "comma", "period"):

        return "Unsupported format type"

    if format_type == "bracket":

        return f"[{currency}] {salary}"

    elif format_type == "comma":

        return f"{currency}, {salary}"

    elif format_type == "period":

        return f"{currency}. {salary}"

