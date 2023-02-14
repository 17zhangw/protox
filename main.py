from tune import tune
from parse_args import parse_cmdline_args
import argparse


if __name__ == "__main__":
    # Tune.
    parser = argparse.ArgumentParser(prog="Mythril")
    args = parse_cmdline_args(parser)
    tune(args)
