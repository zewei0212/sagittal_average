from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

def run_averages(file_input='brain_sample.csv', file_output='brain_average.csv'):
    """
    Calculates the average through the sagittal/horizontal planes (rows).
    Input has rows = sagittal/horizontal planes; columns = coronal planes.
    Result: one average per row.
    """
    planes = np.loadtxt(file_input, dtype=int, delimiter=',')


    averages = planes.mean(axis=1, dtype=float, keepdims=True).T  # shape (1, m)

 
    np.savetxt(file_output, averages, fmt='%.1f', delimiter=',')

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Calculates the average for each sagittal-horizontal plane.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('file_input', nargs='?', default="brain_sample.csv")
    parser.add_argument('--file_output', '-o', default="brain_average.csv")
    args = parser.parse_args()
    run_averages(args.file_input, args.file_output)

