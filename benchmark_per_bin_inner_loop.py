"""
Benchmark the per_bin inner population loop at cosmos25-like scale.

Dimensions:
  n_populations = 111   (11 z-bins × 5 M*-bins × 2 split + 1 foreground)
  n_layers = 112        (n_populations + 1 for foreground in solve)
  n_crop ~ 200_000      (crop-circle pixels for a 7-map science run)
  n_src_per_pop ~ 50    (typical for mid-z bins)
  n_iterations = 50     (cosmos25 bootstrap iterations)

Reports:
  - Total wall time and per-iteration cost breakdown
  - Effect of threadpoolctl.threadpool_limits(limits=1) on DGEMV timing
"""

import time
import numpy as np

# ---- Check threadpoolctl availability ----
try:
    from threadpoolctl import threadpool_limits, threadpool_info
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False
    print("WARNING: threadpoolctl not available — BLAS limiting test skipped")


# ---- Replicate stamp_psf_cropped from test_gram_per_bin.py ----

def make_psf_kernel(fwhm_pix=5.0):
    sigma = fwhm_pix / 2.3548200
    half = int(np.ceil(3.0 * sigma))
    y, x = np.mgrid[-half:half+1, -half:half+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel, half


def stamp_psf_cropped(src_rows, src_cols, psf_kernel, psf_half,
                      flat_to_crop, map_shape, n_crop):
    n_rows, n_cols = map_shape
    layer = np.zeros(n_crop, dtype=np.float64)
    threshold = psf_kernel.max() * 1e-4
    ky, kx = np.where(psf_kernel > threshold)
    kvals = psf_kernel[ky, kx]
    k_dr = ky - psf_half
    k_dc = kx - psf_half
    n_kpix = len(kvals)
    all_rows = src_rows[:, None] + k_dr[None, :]
    all_cols = src_cols[:, None] + k_dc[None, :]
    valid = ((all_rows >= 0) & (all_rows < n_rows)
             & (all_cols >= 0) & (all_cols < n_cols))
    all_flat = np.where(valid, all_rows * n_cols + all_cols, 0)
    all_crop = flat_to_crop[all_flat.ravel()].reshape(all_flat.shape)
    valid &= all_crop >= 0
    v_mask = valid.ravel()
    v_crop_idx = all_crop.ravel()[v_mask]
    v_vals = np.broadcast_to(
        kvals[None, :], (len(src_rows), n_kpix)
    ).ravel()[v_mask]
    np.add.at(layer, v_crop_idx, v_vals)
    return layer


# ---- Synthetic data setup at cosmos25 scale ----

def build_synthetic_data(
    n_populations=111, n_crop=200_000, n_src_per_pop=50,
    map_rows=1000, map_cols=1200, fwhm_pix=5.0, seed=42
):
    rng = np.random.RandomState(seed)
    map_shape = (map_rows, map_cols)
    n_full = map_rows * map_cols
    n_layers = n_populations + 1  # +1 for foreground

    print(f"Building synthetic data: {n_populations} populations, "
          f"{n_crop:,} crop pixels, {n_src_per_pop} src/pop ...")

    # Source positions per population
    src_rows_all = rng.randint(10, map_rows - 10, size=(n_populations, n_src_per_pop))
    src_cols_all = rng.randint(10, map_cols - 10, size=(n_populations, n_src_per_pop))

    # flat_to_crop mapping: pick random n_crop pixels from the map
    crop_flat_idx = rng.choice(n_full, size=n_crop, replace=False)
    crop_flat_idx.sort()
    flat_to_crop = np.full(n_full, -1, dtype=np.int32)
    flat_to_crop[crop_flat_idx] = np.arange(n_crop, dtype=np.int32)

    psf_kernel, psf_half = make_psf_kernel(fwhm_pix)
    kernel_sum = psf_kernel.sum()

    # Build cache (n_layers, n_crop) — use random for speed; math is the same
    cache = rng.randn(n_layers, n_crop).astype(np.float64)

    # Base Gram matrix and h
    obs_vector = rng.randn(n_crop).astype(np.float64)
    G_base = cache @ cache.T                 # (n_layers, n_layers)
    h_base = cache @ obs_vector              # (n_layers,)

    return {
        "map_shape": map_shape,
        "n_full": n_full,
        "n_crop": n_crop,
        "n_layers": n_layers,
        "n_populations": n_populations,
        "src_rows_all": src_rows_all,
        "src_cols_all": src_cols_all,
        "flat_to_crop": flat_to_crop,
        "psf_kernel": psf_kernel,
        "psf_half": psf_half,
        "kernel_sum": kernel_sum,
        "cache": cache,
        "obs_vector": obs_vector,
        "G_base": G_base,
        "h_base": h_base,
    }


# ---- Single inner-loop iteration (one population k, one bootstrap draw) ----

def run_one_iteration(k, idx_A, data, use_identity=True):
    """Run one bootstrap iteration for population k. Returns timing dict.

    use_identity=True  → new: single DGEMV + algebraic identities for B terms
    use_identity=False → old: two DGEMVs + explicit B dot-products
    """
    d = data
    n_layers = d["n_layers"]
    n_crop = d["n_crop"]
    n_full = d["n_full"]
    cache = d["cache"]
    obs_vector = d["obs_vector"]
    G_base = d["G_base"]
    h_base = d["h_base"]
    psf_kernel = d["psf_kernel"]
    psf_half = d["psf_half"]
    flat_to_crop = d["flat_to_crop"]
    map_shape = d["map_shape"]
    kernel_sum = d["kernel_sum"]

    rows_all = d["src_rows_all"][k]
    cols_all = d["src_cols_all"][k]

    t0 = time.perf_counter()

    # 1. PSF stamp
    layer_A = stamp_psf_cropped(
        rows_all[idx_A], cols_all[idx_A],
        psf_kernel, psf_half, flat_to_crop, map_shape, n_crop,
    )
    full_map_mean_A = len(idx_A) * kernel_sum / n_full
    layer_A -= full_map_mean_A

    t1 = time.perf_counter()

    if use_identity:
        # NEW: one DGEMV; derive B terms algebraically
        A_dot_base = cache @ layer_A
        B_dot_base = G_base[:, k] - A_dot_base

        t2 = time.perf_counter()

        A_dot_A = np.dot(layer_A, layer_A)
        h_A = np.dot(layer_A, obs_vector)
        A_dot_B = A_dot_base[k] - A_dot_A
        B_dot_B = G_base[k, k] - 2 * A_dot_base[k] + A_dot_A
        h_B = h_base[k] - h_A
    else:
        # OLD: two DGEMVs + explicit B dot-products
        layer_B = cache[k] - layer_A
        A_dot_base = cache @ layer_A
        B_dot_base = cache @ layer_B

        t2 = time.perf_counter()

        A_dot_A = np.dot(layer_A, layer_A)
        B_dot_B = np.dot(layer_B, layer_B)
        A_dot_B = np.dot(layer_A, layer_B)
        h_A = np.dot(layer_A, obs_vector)
        h_B = np.dot(layer_B, obs_vector)

    t3 = time.perf_counter()

    # Build (n_layers+1)×(n_layers+1) Gram system
    n_new = n_layers + 1
    G_new = np.empty((n_new, n_new))
    h_new = np.empty(n_new)
    old_keep = np.concatenate([np.arange(k), np.arange(k + 1, n_layers)])
    new_keep = np.concatenate([np.arange(k), np.arange(k + 2, n_new)])
    G_new[np.ix_(new_keep, new_keep)] = G_base[np.ix_(old_keep, old_keep)]
    h_new[new_keep] = h_base[old_keep]
    G_new[k, new_keep] = A_dot_base[old_keep]
    G_new[new_keep, k] = A_dot_base[old_keep]
    G_new[k, k] = A_dot_A
    G_new[k, k + 1] = A_dot_B
    G_new[k + 1, k] = A_dot_B
    G_new[k + 1, new_keep] = B_dot_base[old_keep]
    G_new[new_keep, k + 1] = B_dot_base[old_keep]
    G_new[k + 1, k + 1] = B_dot_B
    h_new[k] = h_A
    h_new[k + 1] = h_B

    t4 = time.perf_counter()

    try:
        x_new = np.linalg.solve(G_new, h_new)
    except np.linalg.LinAlgError:
        x_new, _, _, _ = np.linalg.lstsq(G_new, h_new, rcond=None)

    t5 = time.perf_counter()

    return {
        "stamp_ms": (t1 - t0) * 1e3,
        "dgemv_ms": (t2 - t1) * 1e3,
        "dots_ms":  (t3 - t2) * 1e3,
        "gram_ms":  (t4 - t3) * 1e3,
        "solve_ms": (t5 - t4) * 1e3,
        "total_ms": (t5 - t0) * 1e3,
    }


# ---- Benchmark runner ----

def run_benchmark(data, n_iterations=50, split_fraction=0.5, seed=1,
                  blas_limit=None, use_identity=True, label="baseline"):
    n_populations = data["n_populations"]
    rng_seed = seed

    timings = {k: [] for k in ["stamp_ms", "dgemv_ms", "dots_ms", "gram_ms", "solve_ms", "total_ms"]}

    def _inner():
        for k in range(n_populations):
            n_src = data["src_rows_all"].shape[1]
            n_A = int(n_src * split_fraction)
            for iteration in range(n_iterations):
                rng = np.random.RandomState(rng_seed + k * n_iterations + iteration)
                perm = rng.permutation(n_src)
                idx_A = perm[:n_A]
                t = run_one_iteration(k, idx_A, data, use_identity=use_identity)
                for key in timings:
                    timings[key].append(t[key])

    wall_start = time.perf_counter()
    if blas_limit is not None and HAS_THREADPOOLCTL:
        with threadpool_limits(limits=blas_limit, user_api='blas'):
            _inner()
    else:
        _inner()
    wall_total = time.perf_counter() - wall_start

    n_total = n_populations * n_iterations
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  BLAS limit: {blas_limit if blas_limit else 'unrestricted'}")
    print(f"  Total iterations: {n_populations} pops × {n_iterations} iter = {n_total:,}")
    print(f"  Wall time: {wall_total:.1f}s  ({wall_total/n_total*1e3:.2f} ms/iter)")
    print(f"  {'Component':<14} {'mean ms':>9} {'p50 ms':>9} {'p95 ms':>9} {'frac':>7}")
    print(f"  {'-'*52}")
    for key in ["stamp_ms", "dgemv_ms", "dots_ms", "gram_ms", "solve_ms"]:
        arr = np.array(timings[key])
        frac = arr.mean() / np.array(timings["total_ms"]).mean()
        print(f"  {key:<14} {arr.mean():>9.3f} {np.median(arr):>9.3f} "
              f"{np.percentile(arr, 95):>9.3f} {frac:>7.1%}")
    total_arr = np.array(timings["total_ms"])
    print(f"  {'TOTAL':<14} {total_arr.mean():>9.3f} {np.median(total_arr):>9.3f} "
          f"{np.percentile(total_arr, 95):>9.3f}")
    print(f"\n  Speedup estimate (10 workers): {wall_total/10:.1f}s → "
          f"{wall_total/10/60:.1f} min (ignoring overhead)")
    print(f"{'='*60}")


# ---- Main ----

if __name__ == "__main__":
    print("Simstack4 per-bin inner loop benchmark")
    print(f"threadpoolctl available: {HAS_THREADPOOLCTL}")
    if HAS_THREADPOOLCTL:
        info = threadpool_info()
        blas_libs = [x for x in info if x.get('user_api') == 'blas']
        for lib in blas_libs:
            print(f"  BLAS: {lib.get('internal_api','?')} "
                  f"prefix={lib.get('prefix','?')} "
                  f"num_threads={lib.get('num_threads','?')}")

    data = build_synthetic_data(
        n_populations=111,
        n_crop=200_000,
        n_src_per_pop=50,
        map_rows=1000,
        map_cols=1200,
        fwhm_pix=5.0,
    )

    print(f"\ncache shape: {data['cache'].shape}  "
          f"({data['cache'].nbytes / 1e6:.0f} MB)")
    print(f"G_base shape: {data['G_base'].shape}")

    # Warm-up: run 2 populations × 5 iterations to settle caches
    print("\nWarming up ...")
    data_warmup = build_synthetic_data(n_populations=4, n_crop=1000)
    run_benchmark(data_warmup, n_iterations=5, label="warmup (discarded)")

    # Head-to-head: old, new sequential, new threaded
    run_benchmark(data, n_iterations=50, use_identity=False, label="OLD: 2 DGEMVs")
    run_benchmark(data, n_iterations=50, use_identity=True,  label="NEW: 1 DGEMV + identities")

    # ---- Threaded version ----
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    def run_benchmark_threaded(data, n_iterations=50, split_fraction=0.5, seed=1,
                               n_workers=None, label="threaded"):
        n_populations = data["n_populations"]
        n_workers = n_workers or os.cpu_count() or 1
        lock = threading.Lock()
        errors = np.zeros(n_populations)

        def work_k(k):
            n_src = data["src_rows_all"].shape[1]
            n_A = int(n_src * split_fraction)
            samples = []
            for iteration in range(n_iterations):
                rng = np.random.RandomState(seed + k * n_iterations + iteration)
                perm = rng.permutation(n_src)
                idx_A = perm[:n_A]
                t = run_one_iteration(k, idx_A, data, use_identity=True)
                samples.append(t["total_ms"])
            return k, np.std(samples, ddof=1)

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for k, _ in ex.map(work_k, range(n_populations)):
                pass
        wall_total = time.perf_counter() - wall_start

        n_total = n_populations * n_iterations
        seq_ref = 25.2  # from previous single-threaded NEW run
        print(f"\n{'='*60}")
        print(f"  {label}  (n_workers={n_workers})")
        print(f"  Wall time: {wall_total:.1f}s  ({wall_total/n_total*1e3:.2f} ms/task)")
        print(f"  Speedup vs sequential NEW: {seq_ref/wall_total:.2f}×")
        print(f"{'='*60}")

    print(f"\nCPU count: {os.cpu_count()}")
    for nw in [2, 4, os.cpu_count()]:
        run_benchmark_threaded(data, n_iterations=50,
                               n_workers=nw, label=f"Threaded")
