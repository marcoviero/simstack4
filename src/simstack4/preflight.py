"""
Compute-cost preflight estimator for the stacking pipeline.

Before running the expensive PSF-convolution + solve loop, estimate peak
memory and wall-clock time on THIS machine, then let the caller decide
whether to pause for confirmation. Rather than guessing FLOP rates, this:

  - computes EXACT per-map pixel counts (cropped and full) by reusing the
    real, cheap geometry code (`SimstackAlgorithm._compute_crop_geometry`,
    which touches no layer data), so the memory estimate is not a guess;
  - times ONE real PSF convolution (`_create_and_convolve_layer`) and, for
    `bootstrap.method="per_bin"`, ONE real direct-stamp call
    (`_stamp_psf_cropped`) on THIS machine, then scales those measured
    per-call costs by the known population/map/iteration counts.

`bootstrap.method="all_bins"` is a materially different, much heavier code
path (`_stack_single_map_with_bootstrap_splits` rebuilds an UNCROPPED
(2 x populations) layer matrix via a fresh FFT convolution for every single
iteration, vs. `per_bin`'s one-time cropped cache + cheap direct-stamp
iterations) -- the model branches on this explicitly rather than pretending
both methods cost the same.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .algorithm import SimstackAlgorithm
from .config import SimstackConfig
from .populations import PopulationManager
from .sky_maps import SkyMaps
from .utils import format_memory, format_time, memory_usage_gb


@dataclass
class MapEstimate:
    name: str
    n_full_pixels: int
    n_crop_pixels: int
    n_layers: int


@dataclass
class ComputeEstimate:
    n_populations: int
    n_maps: int
    bootstrap_method: str | None
    bootstrap_iterations: int
    per_map: list[MapEstimate]
    peak_memory_bytes: float
    estimated_seconds: float
    available_memory_bytes: float | None
    notes: list[str] = field(default_factory=list)

    def exceeds(
        self, *, memory_fraction: float = 0.9, time_seconds: float = 3600.0
    ) -> bool:
        """True if the estimate crosses either the memory or time budget."""
        mem_flag = (
            self.available_memory_bytes is not None
            and self.peak_memory_bytes > memory_fraction * self.available_memory_bytes
        )
        time_flag = self.estimated_seconds > time_seconds
        return mem_flag or time_flag

    def report(self) -> str:
        lines = [
            f"Populations: {self.n_populations}   Maps: {self.n_maps}",
        ]
        if self.bootstrap_method:
            lines.append(
                f"Bootstrap: {self.bootstrap_method}, "
                f"{self.bootstrap_iterations} iterations"
            )
        else:
            lines.append("Bootstrap: disabled")
        lines.append("")
        for m in self.per_map:
            lines.append(
                f"  {m.name}: {m.n_full_pixels:,} px full, "
                f"{m.n_crop_pixels:,} px cropped, {m.n_layers} layers"
            )
        lines.append("")
        lines.append(f"Estimated peak memory: {format_memory(self.peak_memory_bytes)}")
        if self.available_memory_bytes is not None:
            lines.append(
                f"Available RAM:         {format_memory(self.available_memory_bytes)}"
            )
        lines.append(f"Estimated time:        {format_time(self.estimated_seconds)}")
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


def _map_pixel_counts(
    algo: SimstackAlgorithm, map_name: str, n_populations: int
) -> MapEstimate:
    """Exact full/cropped pixel counts for one map -- geometry only, no convolution."""
    map_data = algo.sky_maps[map_name]
    n_full = map_data.shape[0] * map_data.shape[1]

    if algo.crop_circles:
        flat_indices, _ = algo._compute_crop_geometry(map_name, map_data)
        n_crop = len(flat_indices)
    else:
        signal_flat = (
            map_data.valid_pixel_mask.ravel()
            if map_data.valid_pixel_mask is not None
            else ~np.isnan(map_data.data.ravel())
        )
        n_crop = int(signal_flat.sum())

    has_mask_leak = algo.add_mask_leak and map_data.catalog_mask is not None
    n_layers = (
        n_populations + (1 if algo.add_foreground else 0) + (1 if has_mask_leak else 0)
    )
    return MapEstimate(map_name, n_full, n_crop, n_layers)


def estimate_compute_requirements(
    config: SimstackConfig,
    population_manager: PopulationManager,
    sky_maps: SkyMaps,
) -> ComputeEstimate:
    """
    Estimate peak memory and wall-clock time for running the stacking
    algorithm with the given (already loaded) catalog and maps.

    Requires ``population_manager``/``sky_maps`` to already be loaded (this
    itself is real, unavoidable pipeline work -- not wasted preflight
    overhead -- so the caller can proceed directly into stacking afterward
    without re-loading anything).
    """
    algo = SimstackAlgorithm(config, population_manager, sky_maps)
    n_pops = len(population_manager)
    map_names = list(sky_maps.maps.keys())
    n_maps = len(map_names)
    notes: list[str] = []

    per_map = [_map_pixel_counts(algo, name, n_pops) for name in map_names]
    largest = max(per_map, key=lambda m: m.n_full_pixels)

    sample_pop = next(
        (p for p in population_manager.iter_populations() if p.n_sources > 0), None
    )
    if sample_pop is None:
        raise ValueError(
            "No populations with sources -- cannot estimate compute cost"
        )
    ra, dec = algo._get_coordinates_from_indices(sample_pop.indices[:1])
    map_shape = sky_maps[largest.name].shape

    # ---- Benchmark ONE real FFT convolution on the largest map ----
    t0 = time.perf_counter()
    algo._create_and_convolve_layer(largest.name, ra, dec, map_shape)
    t_conv_ref = time.perf_counter() - t0
    conv_rate = t_conv_ref / largest.n_full_pixels  # seconds per full-map pixel

    bootstrap_enabled = config.error_estimator.bootstrap.enabled
    bootstrap_method = algo.bootstrap_method if bootstrap_enabled else None
    iterations = config.error_estimator.bootstrap.iterations if bootstrap_enabled else 0

    t_conv_all_maps = n_pops * sum(conv_rate * m.n_full_pixels for m in per_map)

    if not bootstrap_enabled:
        t_total = t_conv_all_maps
    elif bootstrap_method == "per_bin":
        # One-time cropped-cache build (above) + cheap direct-stamp iterations.
        map_data = sky_maps[largest.name]
        psf_kernel, psf_half = algo._build_psf_kernel(largest.name)
        me = next(m for m in per_map if m.name == largest.name)
        flat_indices = (
            algo._compute_crop_geometry(largest.name, map_data)[0]
            if algo.crop_circles
            else np.where(
                map_data.valid_pixel_mask.ravel()
                if map_data.valid_pixel_mask is not None
                else ~np.isnan(map_data.data.ravel())
            )[0]
        )
        flat_to_crop = np.full(map_shape[0] * map_shape[1], -1, dtype=np.int32)
        flat_to_crop[flat_indices] = np.arange(me.n_crop_pixels, dtype=np.int32)
        x_pix, y_pix = sky_maps.world_to_pixel(largest.name, ra, dec)
        src_rows = np.round(y_pix).astype(int)
        src_cols = np.round(x_pix).astype(int)

        t0 = time.perf_counter()
        algo._stamp_psf_cropped(
            src_rows, src_cols, psf_kernel, psf_half,
            flat_to_crop, map_shape, me.n_crop_pixels,
        )
        t_stamp_ref = time.perf_counter() - t0

        t_total = t_conv_all_maps + t_stamp_ref * n_pops * iterations * n_maps
    else:  # "all_bins": redoes A+B convolutions, uncropped, every iteration
        t_total = t_conv_all_maps + iterations * 2 * t_conv_all_maps
        notes.append(
            "bootstrap.method='all_bins' redoes the full PSF convolution, "
            "uncropped, for every iteration -- much slower/heavier than "
            "'per_bin'. Consider switching methods if this is unintentional."
        )

    # ---- Memory model ----
    baseline_bytes = memory_usage_gb() * 1024**3  # real RSS: catalog+maps already loaded
    if bootstrap_method == "all_bins":
        n_layers_ab = 2 * n_pops + (1 if algo.add_foreground else 0)
        peak_layer_bytes = max(8 * n_layers_ab * m.n_full_pixels for m in per_map)
        notes.append(
            "all_bins bootstrap builds an UNCROPPED (2x populations) layer "
            "matrix per map -- this dominates peak memory."
        )
    else:
        peak_layer_bytes = max(8 * m.n_layers * m.n_crop_pixels for m in per_map)
        # One uncropped full-map layer is held transiently while the cache
        # is built row-by-row (freed each iteration, so additive not multiplicative).
        peak_layer_bytes += 8 * max(m.n_full_pixels for m in per_map)

    peak_memory_bytes = baseline_bytes + peak_layer_bytes * 1.3  # transient-array buffer

    try:
        import psutil

        available_memory_bytes = psutil.virtual_memory().available
    except ImportError:
        available_memory_bytes = None

    return ComputeEstimate(
        n_populations=n_pops,
        n_maps=n_maps,
        bootstrap_method=bootstrap_method,
        bootstrap_iterations=iterations,
        per_map=per_map,
        peak_memory_bytes=peak_memory_bytes,
        estimated_seconds=t_total,
        available_memory_bytes=available_memory_bytes,
        notes=notes,
    )


def confirm_or_abort(
    estimate: ComputeEstimate,
    *,
    memory_fraction: float = 0.9,
    time_seconds: float = 3600.0,
    assume_yes: bool = False,
) -> bool:
    """
    Print the estimate; if it exceeds the memory/time budget, pause for
    confirmation. Returns True to proceed, False to abort.

    In a non-interactive session (no TTY) without ``assume_yes``, aborts
    for safety rather than blocking on a prompt that will never resolve.
    """
    print(estimate.report())

    if not estimate.exceeds(memory_fraction=memory_fraction, time_seconds=time_seconds):
        return True

    print()
    print(
        f"WARNING: estimated peak memory or run time exceeds the configured "
        f"budget (memory > {memory_fraction:.0%} of available RAM, or "
        f"time > {format_time(time_seconds)})."
    )

    if assume_yes:
        print("--yes passed: continuing without confirmation.")
        return True

    if not sys.stdin.isatty():
        print("Non-interactive session with no --yes flag: aborting for safety.")
        return False

    response = input("Continue anyway? [y/N]: ").strip().lower()
    return response in ("y", "yes")
