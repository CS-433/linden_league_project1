import argparse


def int_percentage(x) -> int:
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "%r not a floating-point literal" % (x,)
        )

    if x <= 0 or x >= 100:
        raise argparse.ArgumentTypeError("%r not in range (0, 100)" % (x,))
    return x


def create_preprocessing_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="preprocesses the BRFSS dataset stored in the directory named data_raw"
    )
    parser.add_argument(
        "--features",
        default="selected",
        choices=["all", "selected", "fraction"],
        nargs="?",
        type=str,
        help="features from the dataset to be used (default: %(default)s)",
    )
    parser.add_argument(
        "--fraction_percentage",
        nargs="?",
        type=int_percentage,
        help="percentage of features to be used (integer between 0 and 100), value should be set if --features is set to fraction",
    )
    return parser
