from typing import AnyStr



def construct_xpath_query(root_element: AnyStr, target_attribute: AnyStr) -> AnyStr:

    """Constructs an XPath query that finds the element with the given target_attribute starting from the root_element.



    Args:

        root_element: The name of the root element from which the search should start.

        target_attribute: The name of the attribute that the target element should have.



    Returns:

        A string that represents the XPath query.

    """

    return f"//{root_element}[@{target_attribute}]"

