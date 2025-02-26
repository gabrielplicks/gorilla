import math



def convert_metric_to_dbm(metric: float) -> float:

    """Converts a signal strength value from a given metric to dBm (decibel-milliwatts).



    Args:

        metric: The signal strength value in an arbitrary metric.



    Raises:

        AssertionError: If the metric is not a positive non-zero number.

    """

    assert metric > 0, "Metric must be a positive non-zero number"

    return 10 * math.log10(metric)

