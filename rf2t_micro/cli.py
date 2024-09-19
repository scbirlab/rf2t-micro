"""Command-line interface for rf2t-micro."""

import sys

from argparse import ArgumentParser, FileType, Namespace
from carabiner import print_err
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp

from .parsers import parse_a3m

__version__ = '0.0.1'

@clicommand(message="Making RosettaFold-2track prediction with the following parameters")
def _run_prediction(args: Namespace) -> None:

    from pandas import DataFrame
    from .predict_msa import Predictor
    
    msa = parse_a3m(args.msa)

    pred = Predictor(model_dir=args.params, use_cpu=args.cpu)
    prediction, cα_coords = pred.predict(msa, args.chain_a_length)
    print_err(prediction.shape)
    columns = [f"A_{i + 1}" for i in range(args.chain_a_length)] + [f"B_{i + 1}" for i in range(msa.shape[-1] - args.chain_a_length)]
    (DataFrame(prediction, 
               index=range(prediction.shape[0]), 
               columns=columns)
     .to_csv(args.output, sep="\t", index=False))

    return None


def main() -> None:
    inputs = CLIOption('msa', 
                       default=sys.stdin,
                       type=FileType('r'), 
                       nargs='?',
                       help='Paired MSA file.')
    plot = CLIOption('--plot', '-p', 
                     type=str,
                     default=None,
                     help='Directory for saving plots. Default: don\'t plot.')
    chain_a_len = CLIOption('--chain-a-length', '-l', 
                     type=int,
                     required=True,
                     help='Number of residues in chain A.')
    output_file = CLIOption('--output', '-o', 
                            default=sys.stdout,
                            type=FileType('w'), 
                            nargs='?',
                            help='Output filename. Default: STDOUT.')
    cpu = CLIOption('--cpu', '-c', 
                    action='store_true',
                    help='Whether to use CPU only. Default: use GPU.')
    params = CLIOption('--params', '-w', 
                       type=str,
                       default=None,
                       help='Path to RosettaFold-2track params file (.npz).')

    run_single = CLICommand('run', 
                            description='Calculate RF2t perdiction for one protein-protein interaction.',
                            main=_run_prediction,
                            options=[inputs, chain_a_len, output_file, plot, cpu, params])

    app = CLIApp("rf2t-micro",
                 version=__version__,
                 description="Make RosettaFold-2track predictions on paired MSAs.",
                 commands=[run_single])

    app.run()
    return None


if __name__ == "__main__":
    main()