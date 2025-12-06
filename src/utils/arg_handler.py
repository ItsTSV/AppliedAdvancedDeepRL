import argparse as ap


def get_args():
    """Parse all training arguments."""
    parser = ap.ArgumentParser(
        description="Reinforcement Learning Trainer"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="What config file to use?"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="online",
        help="Will the training be logged? [online, disabled]"
    )

    args = parser.parse_args()
    if args.config is None:
        raise ValueError("Please, provide a config file using --config argument.")

    return args
