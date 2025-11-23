import numpy as np
import numpy.testing as npt
from pathlib import Path
from sagittal_average.sagittal_brain import run_averages

def test_horizontal_stripe_row_mean(tmp_path: Path):
    
    data_input = np.zeros((20, 20), dtype=int)
    data_input[-1, :] = 1

    in_csv = tmp_path / "brain_sample.csv"
    out_csv = tmp_path / "brain_average.csv"

    np.savetxt(in_csv, data_input, fmt='%d', delimiter=',')

    run_averages(file_input=str(in_csv), file_output=str(out_csv))

    result = np.loadtxt(out_csv, delimiter=',')

    expected = np.zeros(20, dtype=float)
    expected[-1] = 1.0

    npt.assert_array_equal(result, expected)
