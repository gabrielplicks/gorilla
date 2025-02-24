import optparse

import itertools



def scrape_optparse_options(option_parser: optparse.OptionParser) -> dict:

    """

    Returns a dictionary containing all options and their values from an OptionParser object.

    The key is the option name and the value is the option value.

    Args:

        option_parser: The OptionParser object.

    """

    options, args = option_parser.parse_args()

    return dict(itertools.chain(vars(options).items(), [('args', args)]))

