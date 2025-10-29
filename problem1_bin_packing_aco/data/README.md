# Bin Packing Benchmark Data

## Source

OR-Library: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html

## Files Overview

| File         | Type  | Items | Capacity | Instances | Description             |
| ------------ | ----- | ----- | -------- | --------- | ----------------------- |
| binpack1.txt | u120  | 150   | 120      | 20        | Uniform, small capacity |
| binpack2.txt | u250  | 150   | 250      | 20        | Uniform, medium         |
| binpack3.txt | u500  | 150   | 500      | 20        | Uniform, large          |
| binpack4.txt | u1000 | 150   | 1000     | 20        | Uniform, very large     |
| binpack5.txt | t60   | 100   | 60       | 20        | Triplet, small          |
| binpack6.txt | t120  | 100   | 120      | 20        | Triplet, medium         |
| binpack7.txt | t249  | 100   | 249      | 20        | Triplet, large          |
| binpack8.txt | t501  | 100   | 501      | 20        | Triplet, very large     |

**u** = uniform distribution, **t** = triplet distribution

**Note**: Each file contains 20 instances. We use the first instance (\_00) from each file for our experiments.

## Selected Instances for Experiments

We selected **40 instances** exclusively from **uniform distribution** files, using **10 different random seeds per problem size**:

- **u120_00 through u120_09** - 120 items, capacity 150 (small, 10 seeds)
- **u250_00 through u250_09** - 250 items, capacity 150 (medium, 10 seeds)
- **u500_00 through u500_09** - 500 items, capacity 150 (large, 10 seeds)
- **u1000_00 through u1000_09** - 1000 items, capacity 150 (very large, 10 seeds)

**Total: 4 sizes × 10 seeds = 40 benchmark instances**

### Why Uniform Instances Only?

**Triplet instances excluded:** Preliminary analysis revealed that triplet instances (t60, t120, t249, t501) are trivially solved by any greedy heuristic, including worst-case strategies like Worst-Fit. The problem structure naturally produces bins with exactly 3 items at 98-99% capacity regardless of bin selection strategy, providing zero opportunity for demonstrating ACO learning.

**Gap Analysis:**

- Uniform instances: 7-15% gap between random First-Fit and optimal
- Triplet instances: 0% gap (all heuristics achieve optimal)

This selection provides:

- **Problem size diversity**: 120 to 1000 items (scalability analysis)
- **Statistical validity**: Multiple seeds per size (robustness testing)
- **Non-trivial difficulty**: All instances require intelligent optimization
- **Learning opportunity**: Sufficient gap for demonstrating ACO convergence
