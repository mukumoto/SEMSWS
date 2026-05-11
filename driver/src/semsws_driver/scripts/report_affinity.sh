#!/bin/bash
# Bundled with semsws-driver. Each rank dumps its own host/cpu affinity
# to stderr before exec'ing the next command in the chain (typically the
# SEMSWS binary). Enable via:
#   semsws-driver run --report-affinity ...
# or by adding the script's absolute path to `run.binding.rank_wrapper`.

RANK="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${SLURM_PROCID:-${MV2_COMM_WORLD_RANK:-?}}}}"
LRANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${MV2_COMM_WORLD_LOCAL_RANK:-?}}}"
HOST="$(hostname 2>/dev/null)"
AFF="$(taskset -pc $$ 2>/dev/null | awk -F: '{print $2}' | xargs)"
PSR="$(ps -o psr= -p $$ 2>/dev/null | xargs)"

# Print only if there is something to exec; otherwise this is being
# invoked as a probe.
if [ "$#" -gt 0 ]; then
    echo "[AFF] host=$HOST rank=$RANK local=$LRANK pid=$$ aff=$AFF psr=$PSR" >&2
    exec "$@"
else
    echo "[AFF] host=$HOST rank=$RANK local=$LRANK pid=$$ aff=$AFF psr=$PSR"
fi
