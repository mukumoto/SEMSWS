#!/usr/bin/env python3
"""Plot comparison of analytical (ref) vs synthetic (results) pressure seismograms."""

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
ref_dir = os.path.join(script_dir, "ref")
res_dir = os.path.join(script_dir, "results")

stations = ["R1", "R2", "R3", "R4", "R5", "R6"]

fig, axes = plt.subplots(len(stations), 1, figsize=(12, 18), sharex=True)

for i, sta in enumerate(stations):
    ax = axes[i]

    # Load ref (analytic)
    ref_file = os.path.join(ref_dir, f"{sta}.p")
    ref_data = np.loadtxt(ref_file)
    t_ref = ref_data[:, 0]
    p_ref = ref_data[:, 1]

    # Load results (synthetic)
    res_file = os.path.join(res_dir, f"{sta}_0001.p")
    res_data = np.loadtxt(res_file)
    t_syn = res_data[:, 0]
    p_syn = res_data[:, 1]

    # Interpolate to common time if needed
    if len(t_ref) != len(t_syn) or not np.allclose(t_ref, t_syn):
        p_syn_interp = np.interp(t_ref, t_syn, p_syn)
        t = t_ref
        p_s = p_syn_interp
    else:
        t = t_ref
        p_s = p_syn

    residual = p_s - p_ref

    # Plot
    ax.plot(t, p_ref, 'k', lw=1.5, label='Analytic')
    ax.plot(t, p_s, 'r', lw=0.7, label='Synthetic')
    ax.plot(t, residual * 10, 'b', lw=0.5, alpha=0.7, label='Residual x10')

    ax.set_xlim(0, 6)
    ax.set_ylabel(f"{sta}", fontsize=12)
    if i == 0:
        ax.set_title("Pressure (Pa)", fontsize=12)
        ax.legend(fontsize=8, loc='upper right')

    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

fig.suptitle("3D Acoustic Analytic Benchmark: Analytic vs Synthetic", fontsize=14, y=0.995)
fig.supxlabel("Time (s)", fontsize=12)
fig.supylabel("Pressure (Pa)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved comparison.png")
