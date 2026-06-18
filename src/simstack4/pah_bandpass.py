"""
Spectral response curves (bandpasses) for PAH dithered-stacking work.

Provides a small frozen registry of MIPS 24 and MIPS 70 µm relative
spectral response curves, plus helpers for interpolation and rest-frame
coverage. The MIPS 24 arrays are copied verbatim from pah_model.py
(IRSA calibration), which stays frozen as reference code; a Tier-1 test
guards against drift between the two copies. The MIPS 70 arrays come
from the SVO Filter Profile Service (Spitzer/MIPS.70mu), clipped to
non-negative response, peak-normalized, and downsampled to 57 points
(band integral preserved to 0.02%).

Usage:
    from simstack4.pah_bandpass import get_bandpass, BANDPASSES

    bp = get_bandpass("MIPS_24")
    resp = bp.response_at(lam_obs_um)        # 0 outside tabulated range
    lo, hi = bp.rest_coverage(z=2.0)         # rest-frame span probed at z
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

_trapz = np.trapezoid

_N_FINE = 500  # fine-grid points for bandpass integrations


@dataclass(frozen=True)
class Bandpass:
    """Tabulated relative spectral response of a broadband filter."""

    name: str
    lam_um: NDArray[np.float64]  # tabulated observed-frame wavelengths [µm]
    resp: NDArray[np.float64]  # relative response, peak-normalized
    lam_fine: NDArray[np.float64] = field(init=False, repr=False)
    resp_fine: NDArray[np.float64] = field(init=False, repr=False)
    norm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lam_fine = np.linspace(self.lam_um[0], self.lam_um[-1], _N_FINE)
        resp_fine = np.interp(lam_fine, self.lam_um, self.resp, left=0.0, right=0.0)
        object.__setattr__(self, "lam_fine", lam_fine)
        object.__setattr__(self, "resp_fine", resp_fine)
        object.__setattr__(self, "norm", float(_trapz(resp_fine, lam_fine)))

    def response_at(self, lam_obs_um: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolated response at observed wavelengths; 0 outside range."""
        return np.interp(lam_obs_um, self.lam_um, self.resp, left=0.0, right=0.0)

    def rest_coverage(self, z: float) -> tuple[float, float]:
        """Rest-frame wavelength span [µm] probed by this band at redshift z."""
        return float(self.lam_um[0] / (1.0 + z)), float(self.lam_um[-1] / (1.0 + z))

    @property
    def lam_eff(self) -> float:
        """Response-weighted effective wavelength [µm]."""
        return float(_trapz(self.resp_fine * self.lam_fine, self.lam_fine) / self.norm)


# fmt: off
# MIPS 24 µm bandpass (IRSA calibration) — copied verbatim from
# pah_model._MIPS24_LAM / _MIPS24_RESP (frozen reference module).
_MIPS24_LAM = np.array([
    18.005, 19.134, 19.716, 19.944, 20.177, 20.415, 20.577, 20.742, 20.909, 20.993,
    21.079, 21.252, 21.427, 21.606, 21.787, 21.879, 21.972, 22.160, 22.351, 22.545,
    22.743, 22.944, 23.149, 23.358, 23.570, 23.786, 24.006, 24.231, 24.459, 24.692,
    24.930, 25.172, 25.419, 25.670, 25.927, 26.189, 26.456, 26.729, 27.007, 27.292,
    27.582, 27.878, 28.181, 28.491, 28.808, 29.131, 29.462, 29.801, 30.148, 30.865,
    31.618, 32.207,
])
_MIPS24_RESP = np.array([
    0.000237, 0.000644, 0.004662, 0.012914, 0.024468, 0.076447, 0.177656, 0.377777,
    0.735924, 0.813463, 0.857335, 0.907091, 0.957009, 0.984957, 0.997749, 1.000000,
    0.998522, 0.970058, 0.926258, 0.880798, 0.856114, 0.856779, 0.834520, 0.795473,
    0.752764, 0.777653, 0.839877, 0.911819, 0.924876, 0.897806, 0.859096, 0.803216,
    0.736609, 0.649479, 0.558191, 0.486209, 0.428693, 0.381986, 0.326682, 0.261178,
    0.195445, 0.148445, 0.117945, 0.097827, 0.080512, 0.060997, 0.042525, 0.029764,
    0.021807, 0.011002, 0.003471, 0.000727,
])

# MIPS 70 µm bandpass — SVO Filter Profile Service, Spitzer/MIPS.70mu
# (which mirrors the IRSA/SSC calibration); negative baseline values
# clipped to 0, peak-normalized, downsampled 111 → 57 points.
_MIPS70_LAM = np.array([
    49.960, 50.465, 50.980, 51.505, 52.042, 52.589, 53.149, 53.720, 54.304, 54.901,
    55.511, 56.135, 56.773, 57.425, 58.093, 58.776, 59.476, 60.193, 60.927, 61.679,
    62.450, 63.240, 64.051, 64.883, 65.737, 66.613, 67.513, 68.438, 69.389, 70.366,
    71.371, 71.885, 72.406, 73.471, 74.567, 75.697, 76.862, 78.062, 79.302, 80.581,
    81.902, 83.267, 84.678, 86.138, 87.649, 89.214, 90.836, 92.518, 94.264, 96.077,
    97.961, 99.920, 101.959, 104.083, 106.298, 108.609, 111.022,
])
_MIPS70_RESP = np.array([
    0.000000, 0.000865, 0.002934, 0.006083, 0.010670, 0.018442, 0.027425, 0.040147,
    0.060465, 0.090089, 0.129536, 0.153442, 0.169927, 0.214790, 0.263329, 0.294627,
    0.336431, 0.411811, 0.490872, 0.559198, 0.641173, 0.737494, 0.807531, 0.837801,
    0.859973, 0.882606, 0.922826, 0.969391, 0.988399, 0.999355, 0.999344, 1.000000,
    0.991561, 0.971951, 0.937998, 0.882469, 0.781217, 0.669789, 0.558061, 0.456467,
    0.378818, 0.307360, 0.250695, 0.200689, 0.159505, 0.130301, 0.107078, 0.089497,
    0.075730, 0.061335, 0.051832, 0.042113, 0.034532, 0.032969, 0.030667, 0.030972,
    0.029331,
])

# fmt: on

MIPS_24 = Bandpass(name="MIPS_24", lam_um=_MIPS24_LAM, resp=_MIPS24_RESP)
MIPS_70 = Bandpass(name="MIPS_70", lam_um=_MIPS70_LAM, resp=_MIPS70_RESP)

BANDPASSES: dict[str, Bandpass] = {
    "MIPS_24": MIPS_24,
    "MIPS_70": MIPS_70,
}


def get_bandpass(name: str) -> Bandpass:
    """Look up a bandpass by name ("MIPS_24" or "MIPS_70")."""
    try:
        return BANDPASSES[name]
    except KeyError:
        raise KeyError(
            f"Unknown bandpass {name!r}; available: {sorted(BANDPASSES)}"
        ) from None
