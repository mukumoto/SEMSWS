#!/usr/bin/env python3
"""Plot comparison of analytical (ref) vs synthetic (results) seismograms."""

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
ref_dir = os.path.join(script_dir, "ref")
res_dir = os.path.join(script_dir, "results")

stations = ["R1", "R2", "R3", "R4", "R5", "R6"]
comps = ["x", "y", "z"]
comp_labels = ["X (East)", "Y (North)", "Z (Up)"]

fig, axes = plt.subplots(len(stations), len(comps), figsize=(18, 24),
                         sharex=True)

for i, sta in enumerate(stations):
    for j, comp in enumerate(comps):
        ax = axes[i, j]

        # Load ref (analytic)
        ref_file = os.path.join(ref_dir, f"{sta}_{comp}.d")
        ref_data = np.loadtxt(ref_file)
        t_ref = ref_data[:, 0]
        u_ref = ref_data[:, 1]

        # Load results (synthetic)
        res_file = os.path.join(res_dir, f"{sta}_{comp}_0001.d")
        res_data = np.loadtxt(res_file)
        t_syn = res_data[:, 0]
        u_syn = res_data[:, 1]

        # Interpolate to common time if needed
        if len(t_ref) != len(t_syn) or not np.allclose(t_ref, t_syn):
            u_syn_interp = np.interp(t_ref, t_syn, u_syn)
            t = t_ref
            u_s = u_syn_interp
        else:
            t = t_ref
            u_s = u_syn

        residual = u_s - u_ref

        # Plot
        ax.plot(t, u_ref, 'k', lw=1.5, label='Analytic')
        ax.plot(t, u_s, 'r', lw=0.7, label='Synthetic')
        ax.plot(t, residual * 10, 'b', lw=0.5, alpha=0.7, label='Residual x10')

        ax.set_xlim(0, 8)
        if i == 0:
            ax.set_title(comp_labels[j], fontsize=12)
        if j == 0:
            ax.set_ylabel(f"{sta}", fontsize=12)
        if i == 0 and j == 2:
            ax.legend(fontsize=8, loc='upper right')

        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

fig.suptitle("3D Elastic Analytic Benchmark: Analytic vs Synthetic", fontsize=14, y=0.995)
fig.supxlabel("Time (s)", fontsize=12)
fig.supylabel("Displacement (m)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved comparison.png")
