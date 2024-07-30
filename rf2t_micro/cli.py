"""Command-line interface for rf2t-micro."""

import sys

from argparse import ArgumentParser, FileType, Namespace
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp

from .parsers import parse_a3m

__version__ = '0.0.1'

@clicommand(message="Making RosettaFold-2track prediction with the following parameters.")
def _run_prediction(args: Namespace) -> None:

    from .predict_msa import Predictor
    
    msa = parse_a3m(args.msa)

    pred = Predictor(model_dir=args.params, use_cpu=args.cpu)
    pred.predict(msa, args.chain_a_length)
    print(msa.shape)
    with args.output as f:
        f.write(msa)

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
                 description="Screening protein-protein interactions using DCA and AlphaFold2.",
                 commands=[run_single])

    app.run()
    return None


if __name__ == "__main__":
    main()