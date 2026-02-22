# Per-Bin Bootstrap Optimization — Progress Notes

## Problem

With 132 populations × 7 maps × 5 iterations, the old per_bin method:
- Called `_stack_single_map()` 4,620 times
- Each call rebuilt ALL 133 layers via FFT convolution
- Each call solved lstsq on (3.3M × 133) matrix
- Total: ~614K PSF convolutions, 4,620 enormous linear solves
- Result: ~22 min just caching, then hours for iterations

## Solution: Three optimizations

### 1. Gram matrix solve (130× faster linear algebra)
Instead of lstsq on (3.3M × 133), pre-compute:
- G = AᵀA  (133×133 matrix)
- h = Aᵀb  (133 vector)
- Solve G x = h  (microseconds vs seconds)

For each bootstrap iteration, build (134×134) system by replacing
row/col k with A and B rows/columns. Still trivial to solve.

### 2. PSF stamping (1000× faster layer creation)
Instead of FFT convolution on the full map (~7s each), stamp a
discrete PSF kernel directly at source pixel positions (~0.01s).
- Each source touches only ~625 pixels (25×25 PSF kernel)
- 582 sources × 625 pixels = 363K operations
- vs FFT on 3.3M pixels = 100M+ operations

layer_B = cache[k] - layer_A  (zero convolutions for B)

### 3. Sequential map processing (7× less peak memory)
Process one map at a time, free cache before next map.
Peak memory ≈ one map's cache instead of all seven.

## Expected Performance

| Step | Old | New |
|------|-----|-----|
| Cache build | 22 min (same) | 22 min (same — one FFT per pop per map) |
| Per iteration | ~7s (FFT + lstsq) | ~0.08s (stamp + Gram solve) |
| Total iterations | 132 × 5 × 7 = 4620 × 7s ≈ 9 hours | 4620 × 0.08s ≈ 6 min |
| Peak memory | ~8 GB (all maps cached) | ~3.5 GB (one map at a time) |

**Overall: ~9+ hours → ~30 min (cache + iterations)**

## Files Modified

- `algorithm.py` — Replaced `_run_per_bin_error_estimation()`, added:
  - `_build_per_bin_cache()` — builds cropped layers + crop geometry for one map
  - `_build_psf_kernel()` — Gaussian kernel from beam_fwhm_pixels
  - `_stamp_psf_cropped()` — vectorized PSF stamping into cropped pixel buffer

## Test Results

`test_gram_per_bin.py` — 4/4 passing:
- `TestGramMatrixSolveEquivalence` — Gram solve matches lstsq exactly
- `TestPSFStampingAccuracy` — PSF stamp correlates >0.99 with FFT convolution
- `TestPerBinGramVsNaive` — flux_A + flux_B matches between methods (atol=0.02)
- `TestPerBinErrorConsistency` — bootstrap errors are positive, finite, reasonable

## Key Implementation Details

### Mean subtraction
Base layers are mean-subtracted over the full map before cropping.
For PSF-stamped layer_A: mean = n_A / n_full_pixels (since PSF sums to 1).
layer_B = cache[k] - layer_A automatically has correct mean.

### Gram matrix update
When splitting pop k into A/B, the (n_layers+1) system is built by:
1. Copying unchanged block from G_base (all rows/cols except k)
2. Computing A_dot_base = cache @ layer_A (one mat-vec per iteration)
3. B_dot_base = cache @ layer_B
4. Self-products A_dot_A, B_dot_B, A_dot_B
5. Projection h_A = layer_A · obs, h_B = layer_B · obs

### Foreground handling
If add_foreground=True, the cache includes a foreground row (all ones,
cropped). The Gram matrix naturally includes it. Per-bin errors for the
foreground layer are set to 0.
