import hashlib

import importlib



def hash_module(module_name: str) -> str:

    """

    Returns the SHA256 hash of the bytecode of a given module.

    Args:

        module_name: The name of the module to hash.

    """

    try:

        module = importlib.import_module(module_name)

    except ModuleNotFoundError:

        raise ModuleNotFoundError(

            f"Module '{module_name}' not found."

        ) from None  # suppress chained exception

    else:

        bytecode = module.__loader__.get_code(module_name).co_code

        return hashlib.sha256(bytecode).hexdigest()

