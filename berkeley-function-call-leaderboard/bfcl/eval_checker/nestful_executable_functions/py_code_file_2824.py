def get_hash_with_largest_id(hashes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:

    """Returns the hash that contains the key "id" with the largest value.



    Args:

        hashes: A list of hashes (dictionaries).



    Returns:

        The hash with the largest "id" value, or None if the input list is empty.

    """

    if not hashes:

        return None



    return max(hashes, key=lambda hash: hash.get("id", float("-inf")))

