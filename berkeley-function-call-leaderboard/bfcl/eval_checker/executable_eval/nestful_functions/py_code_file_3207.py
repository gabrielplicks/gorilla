import colorsys



def convert_hex_to_hls(hex_string: str) -> tuple:

    """Converts a hex string (RGB) to HLS (Hue, Lightness, Saturation).



    Args:

        hex_string: The hex string representing the RGB color.



    Returns:

        A tuple of floats (H, L, S) representing the HLS color.

    """

    r = int(hex_string[1:3], 16) / 255

    g = int(hex_string[3:5], 16) / 255

    b = int(hex_string[5:7], 16) / 255

    h, l, s = colorsys.rgb_to_hls(r, g, b)

    return (h * 360, l * 100, s * 100)

