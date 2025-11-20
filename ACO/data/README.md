# Benchmark data

All test cases are taken from the OR-Library bin packing set:
<https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html>

## Files in this folder

| File             | Tag   | Capacity | Notes                              |
| ---------------- | ----- | -------- | ---------------------------------- |
| `binpack1.txt`   | u120  | 150      | Uniform instances, 120 items       |
| `binpack2.txt`   | u250  | 150      | Uniform instances, 250 items       |
| `binpack3.txt`   | u500  | 150      | Uniform instances, 500 items       |
| `binpack4.txt`   | u1000 | 150      | Uniform instances, 1000 items      |
| `binpack5-8.txt` | t\*   | 60–501   | Triplet instances (usually easier) |

Each file contains 20 labelled instances (`_00` … `_19`). The ACO experiments focus on the uniform sets (`u*`), typically using the first ten instances of each file (e.g. `u500_00` … `u500_09`). Triplet sets (`t*`) are kept here for completeness but are rarely used because greedy baselines already solve them optimally.

the loader reads whatever files are present.
