from typing import List, Dict



def decode_ldap_data(data: List[Dict[str, bytes]], field_name: str) -> List[str]:

    """Decodes the values of a specified field in a list of LDAP data dictionaries.



    Args:

        data: A list of dictionaries, each representing one LDAP entry.

        field_name: The field name to extract the values for.



    Returns:

        A list of strings containing the decoded values of the specified field.

    """

    decoded_values = []

    for entry in data:

        if field_name in entry:

            decoded_values.append(entry[field_name].decode("utf-8"))

    return decoded_values

