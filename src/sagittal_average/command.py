from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .sagittal_brain import run_averages

def main() -> int:
    parser = ArgumentParser(
        description="Calculates the average for each sagittal-horizontal plane.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('file_input', nargs='?', default="brain_sample.csv",
                        help="Input CSV file with the results from scikit-brain binning algorithm.")
    parser.add_argument('--file_output', '-o', default="brain_average.csv",
                        help="Name of the output CSV file.")
    args = parser.parse_args()
    run_averages(args.file_input, args.file_output)
    return 0
