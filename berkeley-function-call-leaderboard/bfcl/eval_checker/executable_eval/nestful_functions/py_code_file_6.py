from typing import AnyStr



def convert_module_name(module_name: AnyStr) -> AnyStr:

    """Converts a module name in the format `example.module.name` into the format `example_module_name`.



    Args:

        module_name: The module name to be converted.

    """

    return module_name.replace('.', '_')

