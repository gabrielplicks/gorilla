import keyword



def generate_field_names(field_names: list[str], prefix: str) -> list[str]:

    """Generates field names with a prefix and ensures they're valid Python identifiers.



    Args:

        field_names: A list of field names.

        prefix: A string to prefix each field name with.



    Returns:

        A list of valid field names.

    """

    generated_field_names = []

    for field_name in field_names:

        prefixed_name = prefix + field_name

        if keyword.iskeyword(prefixed_name):

            prefixed_name = ''.join(x if x.isalnum() else '_' for x in prefixed_name)

        generated_field_names.append(prefixed_name)

    return generated_field_names

