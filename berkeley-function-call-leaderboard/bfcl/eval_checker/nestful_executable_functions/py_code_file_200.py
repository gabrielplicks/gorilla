def construct_ldap_query(username: str) -> str:

    """

    Constructs a query to search for a user in LDAP given a username.



    Args:

        username: The username to search for.



    Returns:

        A query string in the form of a string, with placeholders for the username and other parameters.

    """

    return f"(&(objectClass=person)(uid={username}))"

