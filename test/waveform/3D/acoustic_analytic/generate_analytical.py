#!/usr/bin/env python3
"""
Generate analytical solution for 3D acoustic full-space benchmark.

Computes pressure at receivers for a point source in a homogeneous
acoustic medium using the 3D Green's function.

== Analytical Derivation ==

1) Standard pressure wave equation:
   d²p/dt² = c² ∇²p + f(x,t)

   For a point source f = A·S(t)·δ(x-x₀), the 3D Green's function gives:
   p(r,t) = A/(4πc²r) · S(t - r/c)

   This is the direct solution where S(t) is the source time function.

2) Displacement potential equation (what SEMSWS solves):
   SEMSWS defines the potential φ through  u = (1/ρ)∇φ  (not u = ∇φ),
   which absorbs density into the potential (φ = ρ × standard potential ψ).

   The PDE solved is:
   (1/κ) d²φ/dt² = ∇·((1/ρ) ∇φ) + (1/κ) A·S(t)·δ(x-x₀)

   After multiplying by κ and using c² = κ/ρ (homogeneous medium):
   d²φ/dt² = c² ∇²φ + A·S(t)·δ(x-x₀)

   Same wave equation form → same Green's function:
   φ(r,t) = A/(4πc²r) · S(t - r/c)

3) Pressure output (SEMSWS convention):
   Since u = (1/ρ)∇φ, the pressure p = -κ ∇·u = -κ/(ρ) ∇²φ.
   From the wave equation: ∇²φ = (1/c²)(d²φ/dt² - source),
   so away from the source: p = -d²φ/dt².

   Therefore:
   p(r,t) = -d²φ/dt² = -A/(4πc²r) · S''(t - r/c - t₀)

   where S'' is the second time derivative of the Ricker wavelet.

   For S(τ) = (1 - 2aτ²)exp(-aτ²)  with  a = π²f₀²:
   S''(τ) = 2a · exp(-aτ²) · (-3 + 12aτ² - 4a²τ⁴)

Outputs:
  - ref/R[1-6].p: Reference pressure seismograms
"""

import os
import numpy as np

# =============================================================================
# Physical parameters (must match config.yaml)
# =============================================================================
VP = 3500.0         # P-wave velocity (m/s)
RHO = 2300.0        # density (kg/m^3)
AMPLITUDE = 1.0e10  # source amplitude

# Ricker wavelet parameters
F0 = 1.0            # center frequency (Hz)
T0_DELAY = 1.5      # time delay / peak time (s)

# Time parameters
DT = 0.005          # time step (s)
NSTEPS = 1200       # number of time steps

# Source location (domain center)
SRC = np.array([12000.0, 12000.0, 12000.0])

# Receiver positions (same as elastic_analytic)
STATIONS = [
    ("R1", np.array([15000.0, 13000.0,  8000.0])),
    ("R2", np.array([13000.0, 15000.0,  8000.0])),
    ("R3", np.array([18000.0, 13000.0,  8000.0])),
    ("R4", np.array([13000.0, 18000.0,  8000.0])),
    ("R5", np.array([18000.0, 13000.0, 10000.0])),
    ("R6", np.array([13000.0, 18000.0, 10000.0])),
]


# =============================================================================
# Ricker wavelet and its second derivative
# =============================================================================
def ricker(t):
    """Ricker wavelet: (1 - 2π²f₀²t²) exp(-π²f₀²t²)"""
    a = np.pi**2 * F0**2
    arg = a * t**2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def ricker_2nd_deriv(t):
    """Analytical second derivative of Ricker wavelet.

    S''(τ) = 2a · exp(-aτ²) · (-3 + 12aτ² - 4a²τ⁴)
    where a = π²f₀².
    """
    a = np.pi**2 * F0**2
    a_t2 = a * t**2
    return 2.0 * a * np.exp(-a_t2) * (-3.0 + 12.0 * a_t2 - 4.0 * a**2 * t**4)


# =============================================================================
# Analytical pressure solution
# =============================================================================
def compute_analytical(r):
    """
    Compute pressure at distance r from point source in 3D acoustic full-space.

    p(r,t) = -A/(4πc²r) · S''(t - r/c - t₀)

    Args:
        r: distance from source (m)

    Returns:
        times: array of time values
        pressure: array of pressure values
    """
    times = np.arange(NSTEPS) * DT
    coeff = -AMPLITUDE / (4.0 * np.pi * VP**2 * r)

    # Retarded time relative to Ricker peak
    tau = times - r / VP - T0_DELAY

    pressure = coeff * ricker_2nd_deriv(tau)

    return times, pressure


# =============================================================================
# Main
# =============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(script_dir, "ref")
    os.makedirs(ref_dir, exist_ok=True)

    print(f"3D Acoustic Analytic Benchmark")
    print(f"Ricker wavelet: f0={F0} Hz, delay={T0_DELAY} s")
    print(f"Material: vp={VP} m/s, rho={RHO} kg/m³")
    print(f"Time: dt={DT}, steps={NSTEPS}, t=[0, {(NSTEPS-1)*DT}] s")
    print(f"Source amplitude: {AMPLITUDE:.1e}")
    print()

    for name, pos in STATIONS:
        r = np.linalg.norm(pos - SRC)
        print(f"Station {name}: position=({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
        print(f"  r={r:.1f} m, P-arrival: {r/VP + T0_DELAY:.3f} s")

        times, pressure = compute_analytical(r)

        fname = os.path.join(ref_dir, f"{name}.p")
        with open(fname, 'w') as f:
            for t, p in zip(times, pressure):
                f.write(f"  {t:14.8E}   {p:14.8E}\n")

        print(f"  Written: {name}.p  (max |p| = {np.max(np.abs(pressure)):.4e})")
        print()

    print("Done. Reference files in ref/")


if __name__ == '__main__':
    main()
