import numpy as np
import numpy.testing as npt
import pathlib
import importlib.util
import sys

# --- Helper: import run_averages(file_input, file_output) from sagittal_brain.py ---
# This makes the test file runnable next to sagittal_brain.py without installing a package.
_MODULE_PATH = pathlib.Path(__file__).parent / "sagittal_brain.py"
spec = importlib.util.spec_from_file_location("sagittal_brain", _MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["sagittal_brain"] = mod
spec.loader.exec_module(mod)  # now we have mod.run_averages

# --- Utilities ---

def write_csv(path: pathlib.Path, arr: np.ndarray) -> None:
    """Write integer input CSV as Charlene's code expects (dtype=int, comma-delimited)."""
    np.savetxt(path, arr.astype(int), fmt='%d', delimiter=',')

# --- Utilities (replace your read_csv; keep write_csv as is) ---

def read_csv(path: pathlib.Path) -> np.ndarray:
    """
    Always read as at-least-2D and coerce to a single-row 2D array.
    This avoids (N,) from loadtxt when the file has a single row.
    """
    arr = np.loadtxt(path, delimiter=',', ndmin=2)  # force >= 2D
    # If someone wrote a column vector by mistake, keep it as (1, N) row vector for comparisons
    if arr.shape[0] != 1 and arr.shape[1] == 1:
        arr = arr.T
    return arr


# --- Tests ---

def test_horizontal_stripe_reveals_row_mean(tmp_path: pathlib.Path):
    """
    Horizontal stripe (last row all ones) should produce a per-row mean:
    expected = [0, 0, ..., 0, 1].

    If the implementation mistakenly averages columns (axis=0),
    it would return ~0.05 for *every* position (for a 20x20 case),
    which our numeric check will reject.
    """
    inp = np.zeros((20, 20), dtype=int)
    inp[-1, :] = 1  # last row is all ones -> row-wise mean is [0,...,0,1]

    in_csv = tmp_path / "brain_sample.csv"
    out_csv = tmp_path / "brain_average.csv"
    write_csv(in_csv, inp)

    mod.run_averages(str(in_csv), str(out_csv))  # run code under test

    out = read_csv(out_csv)  # should be shape (1, 20) row vector
    # Expected numeric row-wise mean: 19 zeros + 1 one
    expected = np.zeros((1, 20))
    expected[0, -1] = 1.0

    npt.assert_allclose(out, expected, atol=1e-8, err_msg="Row-wise mean incorrect for horizontal stripe")

def test_vertical_stripe_catches_axis_swap(tmp_path: pathlib.Path):
    """
    Vertical stripe (last column all ones) is the "mirror" of the previous case.
    For per-row means, each row has exactly one '1' across 20 columns -> mean = 1/20 everywhere.

    If the code mistakenly averages columns (axis=0), it would instead place a 1.0
    at the last *column* and zeros elsewhere, i.e., it would look like the previous test's expected.
    """
    inp = np.zeros((20, 20), dtype=int)
    inp[:, -1] = 1  # last column ones

    in_csv = tmp_path / "brain_sample.csv"
    out_csv = tmp_path / "brain_average.csv"
    write_csv(in_csv, inp)

    mod.run_averages(str(in_csv), str(out_csv))

    # test_vertical_stripe_catches_axis_swap

    out = read_csv(out_csv)

    expected = np.full((1, 20), 0.1, dtype=float)

    np.testing.assert_allclose(
        out, expected, atol=1e-8,
        err_msg="Axis confusion: expected uniform per-row mean saved with one decimal (0.1)")



def test_shape_matches_number_of_rows_with_non_square_input(tmp_path: pathlib.Path):
    """
    Non-square input (10x30) should yield one average per *row* (sagittal/horizontal plane),
    i.e., output length == 10 (kept as a row vector with shape (1, 10)).

    Using a non-square grid prevents false positives where tests only pass on 20x20 by accident.
    """
    rng = np.random.default_rng(0)
    inp = rng.integers(0, 2, size=(10, 30))  # random 0/1 grid, non-square

    in_csv = tmp_path / "brain_sample.csv"
    out_csv = tmp_path / "brain_average.csv"
    write_csv(in_csv, inp)

    mod.run_averages(str(in_csv), str(out_csv))

    out = read_csv(out_csv)
    expected = inp.mean(axis=1, dtype=float)[np.newaxis, :]
    expected = np.round(expected, 1)  
    npt.assert_allclose(out, expected, atol=1e-12)  

def test_transpose_not_accidentally_applied(tmp_path: pathlib.Path):
    """
    Checkerboard-like pattern where column-wise and row-wise means differ,
    and an accidental transpose would swap expectations.

    We make a pattern with two distinct row means to detect transposition errors.
    """
    inp = np.vstack([
        np.zeros((5, 20), dtype=int),   
        np.ones((5, 20), dtype=int)     
    ])  # shape (10, 20)

    in_csv = tmp_path / "brain_sample.csv"
    out_csv = tmp_path / "brain_average.csv"
    write_csv(in_csv, inp)

    mod.run_averages(str(in_csv), str(out_csv))
    out = read_csv(out_csv)

    expected = np.concatenate([np.zeros(5), np.ones(5)])[np.newaxis, :]
    npt.assert_allclose(out, expected, atol=1e-12, err_msg="Unexpected transpose or axis misuse detected")


import numpy as np, subprocess, sys, shutil


m, n = 20, 20
inp = np.zeros((m, n), dtype=int)
inp[:, -1] = 1


np.savetxt("brain_sample.csv", inp, fmt='%d', delimiter=',')

subprocess.run(
    [sys.executable, "sagittal_brain.py", "brain_sample.csv", "-o", "brain_average.csv"],
    check=True
)

arr_in  = np.loadtxt("brain_sample.csv",  delimiter=',', ndmin=2)
arr_out = np.loadtxt("brain_average.csv", delimiter=',', ndmin=2)

print("INPUT  brain_sample.csv  shape:", arr_in.shape)
print(arr_in)
print("\nOUTPUT brain_average.csv shape:", arr_out.shape)
print(arr_out)

