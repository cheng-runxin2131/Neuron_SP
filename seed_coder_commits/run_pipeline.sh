#!/usr/bin/env bash
# run_pipeline.sh — Seed-Coder commit数据采集一键运行
#
# 用法:
#   GITHUB_TOKEN=ghp_xxx bash run_pipeline.sh
#   GITHUB_TOKEN=ghp_xxx bash run_pipeline.sh --small   # 小规模测试(2个org, 每个5repo)
#   GITHUB_TOKEN=ghp_xxx bash run_pipeline.sh --medium  # 中等(10个org, top30)
#   GITHUB_TOKEN=ghp_xxx bash run_pipeline.sh --full    # 全量(所有大厂org)
#
# 断点续传: 直接重跑，自动从上次中断处继续
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 参数 ──
SCALE="${1:---medium}"
case "$SCALE" in
    --small)
        ORGS="nvidia,rapidsai"
        TOP_REPOS=5
        MAX_CONTRIBS=10
        MAX_REPOS_STEP2=20
        MAX_COMMITS=100
        ;;
    --medium)
        ORGS=""  # 用默认列表
        TOP_REPOS=30
        MAX_CONTRIBS=20
        MAX_REPOS_STEP2=500
        MAX_COMMITS=500
        ;;
    --full)
        ORGS=""
        TOP_REPOS=50
        MAX_CONTRIBS=30
        MAX_REPOS_STEP2=0  # 全部
        MAX_COMMITS=1000
        ;;
    *)
        echo "用法: GITHUB_TOKEN=ghp_xxx bash run_pipeline.sh [--small|--medium|--full]"
        exit 1
        ;;
esac

if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "ERROR: 设置 GITHUB_TOKEN"
    echo "  export GITHUB_TOKEN=ghp_xxxx"
    echo "  bash run_pipeline.sh"
    exit 1
fi

echo "============================================"
echo " Seed-Coder Commit数据采集 Pipeline"
echo " Scale: $SCALE"
echo " Time:  $(date)"
echo "============================================"

# ── Step 1: 发现仓库 ──
echo ""
echo ">>> Step 1: 从大厂org出发，通过committer网络扩散发现仓库"
echo ""

STEP1_ARGS="--top-repos-per-org $TOP_REPOS --max-contributors $MAX_CONTRIBS"
if [ -n "$ORGS" ]; then
    STEP1_ARGS="$STEP1_ARGS --orgs $ORGS"
fi
python3 step1_discover_repos.py $STEP1_ARGS
STEP1_EXIT=$?
if [ $STEP1_EXIT -ne 0 ]; then
    echo "[ERROR] Step 1 失败 (exit=$STEP1_EXIT)"
    exit 1
fi

echo ""
echo "Step 1 结果:"
wc -l repos_discovered.jsonl
echo ""

# ── Step 2: Clone + 提取commit ──
echo ">>> Step 2: Clone仓库 + 提取commit数据"
echo ""

STEP2_ARGS="--max-commits-per-repo $MAX_COMMITS"
if [ $MAX_REPOS_STEP2 -gt 0 ]; then
    STEP2_ARGS="$STEP2_ARGS --max-repos $MAX_REPOS_STEP2"
fi
python3 step2_extract_commits.py $STEP2_ARGS
STEP2_EXIT=$?
if [ $STEP2_EXIT -ne 0 ]; then
    echo "[ERROR] Step 2 失败 (exit=$STEP2_EXIT)"
    exit 1
fi

echo ""
echo "Step 2 结果:"
echo "  jsonl文件数: $(ls commits_extracted/*.jsonl 2>/dev/null | wc -l)"
echo "  总样本数: $(cat commits_extracted/*.jsonl 2>/dev/null | wc -l)"
echo ""

# ── Step 3: 去重 ──
echo ">>> Step 3: SHA256 + MinHash去重"
echo ""
python3 step3_dedup_and_stats.py
STEP3_EXIT=$?
if [ $STEP3_EXIT -ne 0 ]; then
    echo "[ERROR] Step 3 失败 (exit=$STEP3_EXIT)"
    exit 1
fi

echo ""
echo "============================================"
echo " Pipeline 完成!"
echo " 最终数据: deduped_commits.jsonl"
echo " 统计信息: dedup_stats.json"
echo " $(date)"
echo "============================================"
