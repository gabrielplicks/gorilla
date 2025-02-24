import numpy as np



def quantize_tensor(tensor: np.ndarray, q: float) -> np.ndarray:

    """Quantizes a tensor of floating-point values to 8-bit integers.

    Args:

        tensor: The input tensor of floating-point values.

        q: The quantization parameter.

    """

    rounded_values = np.round(tensor / q).astype(np.uint8)

    quantized_tensor = np.array(rounded_values)

    return quantized_tensor

