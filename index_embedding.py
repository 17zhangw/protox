from embeddings.gen_index_data import create_datagen_parser
from embeddings.train import create_train_parser
from embeddings.eval_embeddings import create_eval_parser
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MythrilIndexEmbedding")
    subparsers = parser.add_subparsers(help="Subparsers of index embedding.")
    create_datagen_parser(subparsers)
    create_train_parser(subparsers)
    create_eval_parser(subparsers)
    args = parser.parse_args()
    if args.func is not None:
        args.func(args)
