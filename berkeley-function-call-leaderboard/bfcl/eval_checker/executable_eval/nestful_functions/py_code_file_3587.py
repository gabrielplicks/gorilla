import re



def transform_attribute_name(attribute_name: str) -> str:

    """Transforms the input attribute name according to the mapping rules.



    Args:

        attribute_name: The input attribute name to be transformed.



    Returns:

        The transformed attribute name.

    """

    attribute_name = attribute_name.lower().strip()

    if "." in attribute_name:

        attribute_names = attribute_name.split(".")

    else:

        attribute_names = [attribute_name]



    for i, attribute_name in enumerate(attribute_names):

        attribute_name = attribute_name.replace(" ", "_")

        attribute_name = re.sub(r"\d+", lambda m: f"_{m.group(0)}", attribute_name)

        attribute_names[i] = attribute_name



    return "_".join(attribute_names)

