#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
JOB_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/run_square_q2_groot_n15_job.sh"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-/data_all/gzr1/experiment_results/q2_groot_n15_square/${RUN_ID}}"
SEEDS_STRING="${SEEDS:-0 1 2 3 4}"
WANDB_GROUP="${WANDB_GROUP:-square_q2_groot_n15_${RUN_ID}}"
export RUN_ID RESULT_ROOT WANDB_GROUP

mkdir -p "${RESULT_ROOT}"
printf '%s\n' "fixed4 GPU0" "fixed14 GPU1" "reactive GPU2" "ahr GPU3" > "${RESULT_ROOT}/allocation.txt"

run_worker() {
    local method="$1"
    local gpu="$2"
    shift 2
    local seed
    for seed in ${SEEDS_STRING}; do
        GPU="${gpu}" METHOD="${method}" SEED="${seed}" \
            bash "${JOB_SCRIPT}" "$@"
    done
}

run_worker fixed4 0 "$@" & pid0=$!
run_worker fixed14 1 "$@" & pid1=$!
run_worker reactive 2 "$@" & pid2=$!
run_worker ahr 3 "$@" & pid3=$!

status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
wait "${pid2}" || status=1
wait "${pid3}" || status=1

if [[ "${status}" -eq 0 ]]; then
    touch "${RESULT_ROOT}/COMPLETE"
fi

"${RESIDUAL_PYTHON:-/home/gzr1/miniconda3/envs/residual/bin/python}" \
    "${ROOT_DIR}/resfit/rl_finetuning/scripts/summarize_square_q2_groot_n15.py" \
    "${RESULT_ROOT}" | tee "${RESULT_ROOT}/summary.txt" || true

exit "${status}"
