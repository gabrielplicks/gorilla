import re



def device_name_match(device_name: str, pattern: str) -> bool:

    """Checks if a device name matches a specific pattern.



    The pattern can contain wildcards (*) that match any number of characters, including zero.



    Args:

        device_name: The device name to check.

        pattern: The pattern to match against the device name.



    Returns:

        True if the device name matches the pattern, False otherwise.

    """

    pattern = pattern.replace('*', '.*')

    return bool(re.search(pattern, device_name))

