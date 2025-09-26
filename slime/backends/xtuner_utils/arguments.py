import argparse


def parse_args(add_custom_arguments=None):
    parser = argparse.ArgumentParser(description="SLIME")

    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--train-optimizer-steps", type=int, default=16)
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)

    if add_custom_arguments:
        add_custom_arguments(parser)
    args = parser.parse_args()

    if args.load is None:
        args.load = args.hf_checkpoint

    # TODO: only support per token loss now
    args.calculate_per_token_loss = True

    assert args.sp_size == 1, f"sequence parallel not supported yet."

    return args
