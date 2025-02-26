import pkg_resources



def is_package_installed(package_name: str) -> bool:

    """Check if a third-party Python package is installed.



    Args:

        package_name: The name of the package to check.



    Returns:

        True if the package is installed, False otherwise.

    """

    try:

        pkg_resources.get_distribution(package_name)

    except pkg_resources.DistributionNotFound:

        return False

    return True

