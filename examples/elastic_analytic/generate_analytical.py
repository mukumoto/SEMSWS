#!/usr/bin/env python3
"""
Generate analytical solution for 3D elastic full-space benchmark.

Computes displacement Green's function for a double-couple source (Mrp only)
using Aki & Richards (2002) eq. 4.32-4.33, with a Ricker wavelet STF.

Outputs:
  - ref/XX.Z[1-6].FX[XYZ].semd: Reference seismograms

Adapted from SPECFEM3D analytic_solution_elastic benchmark.

Coordinate conventions:
  Aki & Richards: x=North, y=East, z=Down
  SEM-Next:       x=East,  y=North, z=Up (= SPECFEM3D Cartesian)

The Mrp moment tensor component corresponds to:
  A&R: slip on (x1,x2)-plane along x1 -> radiation sin(2*theta)*cos(phi) etc.
  SEM-Next: Mxz (since r=Up=z, p=East=x)
"""

import os
import numpy as np

# =============================================================================
# Physical parameters
# =============================================================================
RHO = 2300.0       # density (kg/m^3)
ALPHA = 2800.0     # P-wave velocity (m/s)
BETA = 1500.0      # S-wave velocity (m/s)
M0 = 1.0e16        # scalar moment (N-m) = 1e23 dyne-cm

# Ricker wavelet parameters (must match config.yaml)
F0 = 1.0            # center frequency (Hz)
T0_DELAY = 1.5      # time delay / peak time (s)

# Time parameters
DT = 0.001          # time step (s)
NSTEPS = 14000      # number of time steps

# SEM-Next domain
SRC_X = 22000.0
SRC_Y = 22000.0
SRC_Z = 22000.0

# Receivers: defined in A&R convention (x_north, y_east, z_down) relative to source
STATIONS = [
    # name,  x_AR(North), y_AR(East), z_AR(Down)
    ("Z1",   1000.0,      3000.0,     4000.0),
    ("Z2",   3000.0,      1000.0,     4000.0),
    ("Z3",   1000.0,      6000.0,     4000.0),
    ("Z4",   6000.0,      1000.0,     4000.0),
    ("Z5",   1000.0,      6000.0,     2000.0),
    ("Z6",   6000.0,      1000.0,     2000.0),
]


# =============================================================================
# Ricker wavelet and its derivative / integral
# =============================================================================
def ricker(t):
    """Ricker wavelet: (1 - 2*pi^2*f0^2*t^2) * exp(-pi^2*f0^2*t^2)
    where t is time relative to peak (t_abs - t0_delay)."""
    arg = np.pi**2 * F0**2 * t**2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def ricker_diff(t, dt):
    """Numerical derivative of Ricker wavelet."""
    return (ricker(t + dt) - ricker(t - dt)) / (2.0 * dt)


def ricker_conv(t_abs, dt, r, alpha, beta):
    """Convolution of Ricker STF between r/alpha and r/beta (near-field term).
    Int_{r/alpha}^{r/beta} stf(t - tau) * tau * dtau
    """
    nmin = int((r / alpha) / dt)
    nmax = int((r / beta) / dt)
    val = 0.0
    for i in range(nmin, nmax + 1):
        tau = i * dt
        t_rel = t_abs - tau - T0_DELAY  # relative to Ricker peak
        val += ricker(t_rel) * tau * dt
    return val


# =============================================================================
# Analytical solution for one receiver
# =============================================================================
def compute_analytical(x_north, y_east, z_down):
    """
    Compute displacement for Mrp double-couple in homogeneous full-space.

    Args:
        x_north, y_east, z_down: receiver position in A&R convention
                                  (relative to source at origin)

    Returns:
        times: array of time values
        u_east, u_north, u_up: displacement in SEM-Next convention
    """
    # Distance and spherical coordinates (A&R: x=N, y=E, z=Down)
    r = np.sqrt(x_north**2 + y_east**2 + z_down**2)
    theta = np.arccos(z_down / r)
    phi = np.arctan2(y_east, x_north)

    print(f"  r={r:.1f} m, theta={np.degrees(theta):.1f} deg, phi={np.degrees(phi):.1f} deg")
    print(f"  P-arrival: {r/ALPHA:.3f} s, S-arrival: {r/BETA:.3f} s")

    # Time array (simulation starts at t=0)
    n_samples = NSTEPS
    times = np.zeros(n_samples)
    u_sph = np.zeros((n_samples, 3))  # r, theta, phi

    # Amplitude coefficients (Aki & Richards eq. 4.32)
    cn  = M0 / (4.0 * np.pi * RHO * r**4)
    cip = M0 / (4.0 * np.pi * RHO * ALPHA**2 * r**2)
    cis = M0 / (4.0 * np.pi * RHO * BETA**2 * r**2)
    cfp = M0 / (4.0 * np.pi * RHO * ALPHA**3 * r)
    cfs = M0 / (4.0 * np.pi * RHO * BETA**3 * r)

    # Radiation pattern coefficients (eq. 4.33) for Mrp
    an = np.array([
        9.0 * np.sin(2*theta) * np.cos(phi),
        -6.0 * np.cos(2*theta) * np.cos(phi),
        6.0 * np.cos(theta) * np.sin(phi)
    ])
    aip = np.array([
        4.0 * np.sin(2*theta) * np.cos(phi),
        -2.0 * np.cos(2*theta) * np.cos(phi),
        2.0 * np.cos(theta) * np.sin(phi)
    ])
    ais = np.array([
        -3.0 * np.sin(2*theta) * np.cos(phi),
        3.0 * np.cos(2*theta) * np.cos(phi),
        -3.0 * np.cos(theta) * np.sin(phi)
    ])
    afp = np.array([
        np.sin(2*theta) * np.cos(phi),
        0.0,
        0.0
    ])
    afs = np.array([
        0.0,
        np.cos(2*theta) * np.cos(phi),
        -np.cos(theta) * np.sin(phi)
    ])

    # Time loop
    for idx in range(n_samples):
        t_abs = idx * DT
        times[idx] = t_abs

        # Ricker STF evaluated at retarded times (relative to peak)
        t_p = t_abs - r/ALPHA - T0_DELAY
        t_s = t_abs - r/BETA - T0_DELAY

        stf_p = ricker(t_p)
        stf_s = ricker(t_s)
        stf_p_diff = ricker_diff(t_p, DT)
        stf_s_diff = ricker_diff(t_s, DT)
        stf_conv = ricker_conv(t_abs, DT, r, ALPHA, BETA)

        # Displacement in spherical coordinates (r, theta, phi)
        for i in range(3):
            nf = cn * an[i] * stf_conv
            ip = cip * aip[i] * stf_p + cis * ais[i] * stf_s
            ff = cfp * afp[i] * stf_p_diff + cfs * afs[i] * stf_s_diff
            u_sph[idx, i] = nf + ip + ff

    # Convert spherical (r, theta, phi) to A&R Cartesian (x=N, y=E, z=Down)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    u_north = u_sph[:, 0] * st * cp + u_sph[:, 1] * ct * cp - u_sph[:, 2] * sp
    u_east  = u_sph[:, 0] * st * sp + u_sph[:, 1] * ct * sp + u_sph[:, 2] * cp
    u_down  = u_sph[:, 0] * ct      - u_sph[:, 1] * st

    # Convert to SEM-Next convention: x=East, y=North, z=Up
    u_x = u_east
    u_y = u_north
    u_z = -u_down  # Up = -Down

    return times, u_x, u_y, u_z


# =============================================================================
# Main
# =============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(script_dir, "ref")
    os.makedirs(ref_dir, exist_ok=True)

    print(f"Ricker wavelet: f0={F0} Hz, delay={T0_DELAY} s")
    print(f"Time: dt={DT}, steps={NSTEPS}, t=[0, {(NSTEPS-1)*DT}] s")
    print()

    # Generate analytical seismograms for each station
    for name, x_n, y_e, z_d in STATIONS:
        print(f"Station {name}: A&R (x_N={x_n}, y_E={y_e}, z_D={z_d})")

        # SEM-Next position
        sem_x = SRC_X + y_e   # East offset
        sem_y = SRC_Y + x_n   # North offset
        sem_z = SRC_Z - z_d   # Up
        print(f"  SEM-Next position: ({sem_x}, {sem_y}, {sem_z})")

        times, u_x, u_y, u_z = compute_analytical(x_n, y_e, z_d)

        # Write reference files in SPECFEM ASCII format: time amplitude
        for comp, data in [("FXX", u_x), ("FXY", u_y), ("FXZ", u_z)]:
            fname = os.path.join(ref_dir, f"XX.{name}.{comp}.semd")
            with open(fname, 'w') as f:
                for t, u in zip(times, data):
                    f.write(f"  {t:14.8E}   {u:14.8E}\n")

        print(f"  Written: XX.{name}.FX[XYZ].semd")
        print()

    print("Done. Reference files in ref/")
    print()

    # Print receiver locations for config.yaml
    print("SEM-Next receiver locations for config.yaml:")
    for name, x_n, y_e, z_d in STATIONS:
        sem_x = SRC_X + y_e
        sem_y = SRC_Y + x_n
        sem_z = SRC_Z - z_d
        print(f'    - name: "{name}"')
        print(f'      location: [{sem_x:.1f}, {sem_y:.1f}, {sem_z:.1f}]')


if __name__ == '__main__':
    main()
