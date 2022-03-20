"""Parameter parser to set the model hyperparameters."""

import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the partial NCI1 graph dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """
    parser = argparse.ArgumentParser(description="Run Graph2Vec.")

    parser.add_argument("--input-path",
                        nargs="?",
                        default="./dataset/",
                        help="Input folder with jsons.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="./features/nci1.csv",
                        help="Embeddings path.")

    parser.add_argument("--ground-truth",
                        nargs="+",
                        help="Ground truth event time steps.")

    return parser.parse_args()
