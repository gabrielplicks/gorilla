import numpy as np



def generate_numpy_boilerplate(name: str, dtype: np.dtype, shape: tuple) -> str:

    """Generates a boilerplate numpy code snippet for a given variable name, data type, and shape (dimensions).



    Args:

        name: The name of the variable.

        dtype: The data type of the variable.

        shape: The shape (dimensions) of the variable.

    """

    code = f"import numpy as np\n{name} = np.ndarray(shape={shape}, dtype={dtype})"



    return code

