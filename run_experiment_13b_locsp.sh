#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
eval "$(conda shell.bash hook)"
conda activate walking3
set -u

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DESLOC_SP_A2A_TIMEOUT_MS=120000

RESULTS_DIR="./desloc_results"
mkdir -p "$RESULTS_DIR"
NGPU=3
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs_13b_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " LOC+SP 13B — 2×A6000 + 1×H100 NVL"
echo " $(date)"
echo " NGPU: $NGPU"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================"

PORT=29600

run_exp() {
    local NAME="$1"; local MODEL="$2"; local KX="$3"; local METHODS="$4"
    local STEPS="$5"; local BATCH="${6:-1}"; local GRAD_ACCUM="${7:-8}"
    local EXTRA="${8:-}"
    local KU=$((KX * 3)); local KV=$((KX * 6))
    if [ "$KX" -eq 1 ]; then KU=1; KV=1; fi

    echo ""
    echo ">>> [$(date +%H:%M:%S)] $NAME | model=$MODEL Kx=$KX methods=$METHODS steps=$STEPS batch=$BATCH×$GRAD_ACCUM $EXTRA"

    CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
        --nproc_per_node=$NGPU --master_addr=127.0.0.1 --master_port=$PORT \
        REAL_GPU_BENCHMARK.py \
        --model_size "$MODEL" --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
        --max_steps "$STEPS" --Kx "$KX" --Ku "$KU" --Kv "$KV" \
        --methods $METHODS --output "$RESULTS_DIR" \
        $EXTRA \
        2>&1 | tee "$LOG_DIR/${NAME}.log"

    local EC=${PIPESTATUS[0]}
    if [ $EC -ne 0 ]; then echo "!!! $NAME FAILED (exit=$EC)"; else echo "<<< $NAME OK"; fi
    PORT=$((PORT + 1)); sleep 5
}

echo "===== Phase 13a: 13B DESLOC+SP baseline (Kx=1 = DDP-equivalent) ====="
PYTHONHASHSEED=13 run_exp "p13a_desloc_13B_Kx1" "13B" 1 "DESLOC" 200 1 8 "--cpu_offload --use_ac --use_autosp"

echo "===== Phase 13b: 13B Kx ablation (200 steps) ====="
for KX in 16 32 64; do
    PYTHONHASHSEED=13 run_exp "p13b_sp_desloc_13B_Kx${KX}" "13B" "$KX" "DESLOC" 200 1 8 "--cpu_offload --use_ac --use_autosp"
done

echo "===== Phase 13c: 13B extended (1536 steps, Kx=32) ====="
PYTHONHASHSEED=13 run_exp "p13c_sp_desloc_13B_Kx32_long" "13B" 32 "DESLOC" 1536 1 8 "--cpu_offload --use_ac --use_autosp"

echo "===== Phase 13d: 13B Nesterov vs Avg ====="
PYTHONHASHSEED=13 run_exp "p13d_sp_nesterov_13B" "13B" 32 "DESLOC" 200 1 8 "--cpu_offload --use_ac --use_autosp --outer_optimizer nesterov --outer_momentum 0.9"
PYTHONHASHSEED=13 run_exp "p13d_sp_avg_13B" "13B" 32 "DESLOC" 200 1 8 "--cpu_offload --use_ac --use_autosp"

echo "================================================================"
echo " 13B LOC+SP done — $(date)"
echo " JSON: $(ls -1 $RESULTS_DIR/*.json 2>/dev/null | wc -l)"
echo "================================================================"
