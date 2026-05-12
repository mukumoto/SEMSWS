#!/bin/bash
# JAMSTEC Text Mesh Example — Full Workflow
# Generates mesh, runs acoustic simulation with Dirichlet BC on letter boundaries,
# and creates backward-in-time animation.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

HOHQMESH="${HOHQMESH:-$HOME/program_ubuntu/HOHQMesh/HOHQMesh}"
SEM_EXE="${SEM_EXE:-../../../build/src/semsws}"
NP="${NP:-4}"

echo "=== Step 1: Generate mesh control file from font contours ==="
python3 generate_mesh.py --output-dir . \
    --domain-w 14000 --domain-h 4000 --letter-height 2200 \
    --element-size 100 --smooth-sigma 1.6 --smooth-passes 1 \
    --n-points 120 --font "URW Bookman"

echo "=== Step 2: Generate mesh with HOHQMesh ==="
"$HOHQMESH" -f jamstec.control

echo "=== Step 3: Convert ABAQUS -> Gmsh 2.2 (with boundary attributes) ==="
python3 convert_mesh.py jamstec.inp jamstec.msh

echo "=== Step 4: Plot mesh ==="
python3 plot_mesh.py jamstec.inp

echo "=== Step 5: Run simulation ==="
mpirun -np "$NP" "$SEM_EXE" --config config.yaml

echo "=== Step 6: Create animation ==="
python3 make_animation.py

echo "=== Done ==="
