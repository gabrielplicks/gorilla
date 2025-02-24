from typing import List



def parse_module_name(module_name: str) -> List[str]:

    """Parses the module name of a dot-chained Python module.



    Args:

        module_name: The dot-chained Python module name.



    Returns:

        A list of strings corresponding to the module names in the chain.

    """

    try:

        modules = module_name.split(".")

        return modules

    except Exception as e:

        print(f"Failed to parse module name: {e}")

        return []

