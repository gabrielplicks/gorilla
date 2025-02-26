from typing import Union



def build_span_string(n: Union[int, float]) -> str:

    """

    Builds an HTML string containing the numbers from 1 to `n` enclosed in a `<span>` element.



    Args:

        n: The number of `<span>` elements to generate.



    Returns:

        An HTML string containing the numbers from 1 to `n` enclosed in a `<span>` element.

    """

    assert isinstance(n, int) and n > 0, "n must be a positive integer"



    span_string = "<span>{}</span>"

    html_string = ""



    for i in range(1, n + 1):

        html_string += span_string.format(i)



    return html_string

