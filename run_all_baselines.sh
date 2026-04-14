#!/bin/bash
# 按序运行5个 baseline 训练脚本
# 用法：
#   bash run_all_baselines.sh          # 默认运行 cost 任务
#   bash run_all_baselines.sh cost     # 预测 Execution Time
#   bash run_all_baselines.sh card     # 预测 Cardinality

set -e  # 任意一步失败则退出

TASK=${1:-cost}   # 默认 cost，可传入 card

PYTHON=/Users/yyh/miniforge3/envs/leon_cypher/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Running all baselines: TASK=${TASK}"
echo "========================================"

# 临时修改各脚本的 TASK 变量，运行完后恢复
run_with_task() {
    local script="$1"
    local default_task="$2"   # 该脚本的默认 TASK（用于恢复）

    echo ""
    echo "----------------------------------------"
    echo "  [$(date '+%H:%M:%S')] Starting: $script  (TASK=${TASK})"
    echo "----------------------------------------"

    # 用 sed 临时替换 TASK 值
    sed -i.bak "s/^TASK = '.*'/TASK = '${TASK}'/" "${SCRIPT_DIR}/${script}"

    ${PYTHON} "${SCRIPT_DIR}/${script}"
    local exit_code=$?

    # 恢复原始 TASK 值
    sed -i.bak "s/^TASK = '.*'/TASK = '${default_task}'/" "${SCRIPT_DIR}/${script}"
    rm -f "${SCRIPT_DIR}/${script}.bak"

    if [ $exit_code -ne 0 ]; then
        echo "  [ERROR] ${script} failed with exit code ${exit_code}"
        exit $exit_code
    fi

    echo "  [$(date '+%H:%M:%S')] Finished: $script"
}

cd "${SCRIPT_DIR}"

run_with_task "TrainingV1_tata_baseline.py"   "cost"
run_with_task "TrainingV1_bao_baseline.py"    "cost"
run_with_task "TrainingV1_leon_baseline.py"   "cost"
run_with_task "TrainingV1_mscn_baseline.py"   "card"   # MSCN 默认 card
run_with_task "TrainingV1_qppnet_baseline.py" "cost"

echo ""
echo "========================================"
echo "  All baselines finished! TASK=${TASK}"
echo "========================================"
