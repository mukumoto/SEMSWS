#!/bin/bash
# ============================================================
# 3D Elastic 2-Layer Benchmark — elastic variant
#
# Usage:
#   bash run_elastic.sh [NPROCS]       # default NPROCS=4
#
# Prereqs:
#   - mesh_layered_3d.exo already generated (coreform_cubit -nographics -input mesh_layered_3d.jou)
#   - SEM-Next build at ../../build/src (refine_mesh, evaluate_mesh,
#     test_forward_simulation)
# ============================================================

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
progdir="${REPO_ROOT}/build/src"
plotdir="${REPO_ROOT}/scripts/plot"
NP="${1:-4}"

cd "${HERE}"

echo "=========================================="
echo "3D Elastic 2-Layer Benchmark (CPU)"
echo "NP=${NP}"
echo "Started at: $(date)"
echo "=========================================="

# Step 1: Refine mesh (writes mesh/)
echo ""
echo "Step 1: Refine mesh"
mpirun -np "${NP}" "${progdir}/refine_mesh" --config config_elastic.yaml --output-dir mesh

# Step 2: Evaluate refined mesh (optional diagnostics)
echo ""
echo "Step 2: Evaluate mesh"
mpirun -np "${NP}" "${progdir}/evaluate_mesh" --config config_elastic_sim.yaml --histogram eval

# Step 3: Forward simulation
echo ""
echo "Step 3: Run simulation"
mpirun -np "${NP}" "${progdir}/test_forward_simulation" --config config_elastic_sim.yaml

# Step 4: Plot (optional; needs GMT / pyvista etc.)
echo ""
echo "Step 4: Plot results (optional)"
python3 "${plotdir}/plot_wavefield_3d.py" --results-dir ./results || echo "  (plot_wavefield_3d failed — need matplotlib + numpy)"
python3 "${plotdir}/plot_3d_slices.py"    --results-dir ./results --type both || echo "  (plot_3d_slices failed — install pyvista to enable)"

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
