import hashlib



def check_md5sum(file_path: str, md5_sum: str) -> bool:

    """Checks if the MD5 hash of a file matches a given MD5 sum.



    Args:

        file_path: The path to the file.

        md5_sum: The MD5 sum in hexadecimal format.



    Returns:

        True if the MD5 hash of the file matches the given MD5 sum, False otherwise.

    """

    with open(file_path, 'rb') as file:

        file_content = file.read()

        md5_hash = hashlib.md5(file_content)  # Compute MD5 hash of the file content

        computed_md5_sum = md5_hash.hexdigest()  # Get the hexadecimal representation of the hash

        match = computed_md5_sum == md5_sum  # Compare the computed MD5 hash with the given MD5 sum

        return match

