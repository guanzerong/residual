#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_resvit_depth_stable.sh"

cd "${ROOT_DIR}"

if [ "$#" -gt 0 ]; then
    GPUS=("$@")
else
    GPUS=(0 1 2)
fi

if [ "${#GPUS[@]}" -lt 3 ]; then
    echo "Usage: $0 GPU0 GPU1 GPU2 [GPU3 GPU4 GPU5]"
    echo "Example, 3 GPUs sequential paired: $0 0 3 5"
    echo "Example, 6 GPUs fully parallel paired: PARALLEL_NODEPTH=1 $0 0 1 2 3 4 5"
    exit 2
fi

SEEDS=(1 2 3)
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_localdepth_paired_3seed}"
RUN_NODEPTH="${RUN_NODEPTH:-1}"
PARALLEL_NODEPTH="${PARALLEL_NODEPTH:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/square_localdepth_paired_$(date +%Y%m%d_%H%M%S)}"

if [ "${PARALLEL_NODEPTH}" = "1" ] && [ "${RUN_NODEPTH}" = "1" ] && [ "${#GPUS[@]}" -lt 6 ]; then
    echo "PARALLEL_NODEPTH=1 with RUN_NODEPTH=1 needs 6 GPUs to avoid placing two jobs on one GPU." >&2
    echo "If you intentionally want to oversubscribe 3 GPUs, set ALLOW_GPU_OVERSUBSCRIBE=1." >&2
    if [ "${ALLOW_GPU_OVERSUBSCRIBE:-0}" != "1" ]; then
        exit 2
    fi
fi

mkdir -p "${LOG_DIR}"

run_condition() {
    local gpu="$1"
    local seed="$2"
    local condition="$3"
    local log_file="${LOG_DIR}/${condition}_seed${seed}_gpu${gpu}.log"

    echo "[$(date '+%F %T')] starting condition=${condition} seed=${seed} gpu=${gpu}" | tee -a "${LOG_DIR}/launcher.log"

    if [ "${condition}" = "stabledepth" ]; then
        CUDA_VISIBLE_DEVICES="${gpu}" bash "${BASE_SCRIPT}" \
            seed="${seed}" \
            wandb.name="square_macro4710_stabledepth_seed${seed}" \
            wandb.group="${WANDB_GROUP}" \
            wandb.notes="paired_stabledepth_vs_nodepth_seed${seed}" \
            wandb.mode=online \
            >"${log_file}" 2>&1
    elif [ "${condition}" = "nodepth" ]; then
        CUDA_VISIBLE_DEVICES="${gpu}" bash "${BASE_SCRIPT}" \
            seed="${seed}" \
            agent.depth_anything_v2_patch_state.enabled=false \
            agent.depth_anything_v2_patch_state.token_dropout=0.0 \
            agent.depth_anything_v2_patch_state.token_scale_init=1.0 \
            wandb.name="square_macro4710_nodepth_control_seed${seed}" \
            wandb.group="${WANDB_GROUP}" \
            wandb.notes="paired_stabledepth_vs_nodepth_seed${seed}" \
            wandb.mode=online \
            >"${log_file}" 2>&1
    else
        echo "Unknown condition: ${condition}" >&2
        exit 2
    fi

    echo "[$(date '+%F %T')] finished condition=${condition} seed=${seed} gpu=${gpu}" | tee -a "${LOG_DIR}/launcher.log"
}

if [ "${PARALLEL_NODEPTH}" = "1" ]; then
    for idx in 0 1 2; do
        seed="${SEEDS[$idx]}"
        run_condition "${GPUS[$idx]}" "${seed}" stabledepth &
        if [ "${RUN_NODEPTH}" = "1" ]; then
            nodepth_gpu_idx=$((idx + 3))
            if [ "${nodepth_gpu_idx}" -lt "${#GPUS[@]}" ]; then
                nodepth_gpu="${GPUS[$nodepth_gpu_idx]}"
            else
                nodepth_gpu="${GPUS[$idx]}"
            fi
            run_condition "${nodepth_gpu}" "${seed}" nodepth &
        fi
    done
else
    for idx in 0 1 2; do
        gpu="${GPUS[$idx]}"
        seed="${SEEDS[$idx]}"
        (
            run_condition "${gpu}" "${seed}" stabledepth
            if [ "${RUN_NODEPTH}" = "1" ]; then
                run_condition "${gpu}" "${seed}" nodepth
            fi
        ) &
    done
fi

wait

echo "All paired local-depth runs finished. Logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/launcher.log"
