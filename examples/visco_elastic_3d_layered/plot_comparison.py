#!/usr/bin/env python3
"""Plot comparison of SPECFEM3D vs SEMSWS seismograms for paper."""

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
res_dir = os.path.join(script_dir, "semsws_run", "results")
ref_dir = os.path.join(script_dir, "specfem_run", "OUTPUT_FILES")

stations = ["R01", "R02", "R03"]
comps = ["x", "y", "z"]
comp_labels = ["X", "Y", "Z"]
spf_comp = {"x": "FXX", "y": "FXY", "z": "FXZ"}

fig, axes = plt.subplots(len(stations), len(comps), figsize=(12, 7), sharex=True)

for i, sta in enumerate(stations):
    for j, comp in enumerate(comps):
        ax = axes[i, j]

        # Load SPECFEM3D (reference)
        ref_data = np.loadtxt(os.path.join(ref_dir, f"XX.{sta}.{spf_comp[comp]}.semd"))
        t_ref, u_ref = ref_data[:, 0], ref_data[:, 1]

        # Load SEMSWS (results)
        res_data = np.loadtxt(os.path.join(res_dir, f"{sta}_{comp}_0001.d"))
        t_syn, u_syn = res_data[:, 0], res_data[:, 1]

        # Interpolate to common time if needed
        if len(t_ref) != len(t_syn) or not np.allclose(t_ref, t_syn):
            u_syn = np.interp(t_ref, t_syn, u_syn)
        t = t_ref

        residual = u_syn - u_ref

        # Normalize by scale factor from reference solution
        max_val = np.max(np.abs(u_ref))
        if max_val > 1e-30:
            exp = int(np.floor(np.log10(max_val)))
            scale = 10.0 ** exp
        else:
            scale = 1.0
            exp = 0

        ax.plot(t, u_ref / scale, 'k', lw=1.5, label='SPECFEM3D')
        ax.plot(t, u_syn / scale, 'r', lw=0.8, linestyle='--', label='SEMSWS')
        ax.plot(t, residual * 10 / scale, 'b', lw=0.5, alpha=0.7, label='Residual ×10')

        # Scale label at top-left
        ax.text(0.02, 0.95, f'×1e{exp}', transform=ax.transAxes,
                fontsize=8, ha='left', va='top')

        if i == 0:
            ax.set_title(comp_labels[j], fontsize=12)
        if j == 0:
            ax.set_ylabel(f'{sta}', fontsize=11)
        if i == 0 and j == 2:
            ax.legend(fontsize=8, loc='upper right')
        if i == len(stations) - 1:
            ax.set_xlabel('Time (s)', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "comparison.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, "comparison.pdf"), bbox_inches='tight')
plt.close()
print("Saved: comparison.png, comparison.pdf")
