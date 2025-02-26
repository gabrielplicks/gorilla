from typing import Dict



def get_xpath_expr(tag_name: str, **attrs: Dict[str, str]) -> str:

    """Constructs an XPath expression to match an element with a specific tag name and attributes.



    Args:

        tag_name: The tag name of the element.

        attrs: The attributes of the element.

    """

    attr_conditions = []

    for attr_name, attr_val in attrs.items():

        attr_conditions.append(f'@{attr_name}="{attr_val}"')

    attr_condition_str = " and ".join(attr_conditions)

    return f'//{tag_name}[{attr_condition_str}]'

