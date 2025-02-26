import math



def convert_voltage_to_dbm(voltage: float) -> float:

    """Converts a signal from voltage to dBm using the formula $P_{dBm} = 10 \cdot \log_{10}(P_{dBm})$.



    Args:

        voltage: The input signal in either dBm or voltage.

    """

    dbm = 10 * math.log10(voltage)

    return dbm

