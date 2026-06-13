#!/usr/bin/env bash
# extract_batch.sh — 对一个batch的仓库列表clone+提取commit
# 用法: bash extract_batch.sh batch_000.txt
# 输入: batch文件(每行一个 owner/repo)
# 输出: commits_batch_NNN/ 目录下每个repo一个jsonl
set -uo pipefail

BATCH_FILE="${1:?用法: bash extract_batch.sh batch_XXX.txt}"
BATCH_ID=$(basename "$BATCH_FILE" .txt)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="commits_${BATCH_ID}"
BARE_DIR="repos_bare"
mkdir -p "$OUT_DIR" "$BARE_DIR"

TOTAL=$(wc -l < "$BATCH_FILE")
echo "=== extract_batch: $BATCH_ID ($TOTAL repos) ==="

IDX=0
while IFS= read -r REPO; do
    [ -z "$REPO" ] && continue
    IDX=$((IDX+1))
    SAFE=$(echo "$REPO" | tr '/' '__')
    BARE_PATH="${BARE_DIR}/${SAFE}.git"
    OUT_FILE="${OUT_DIR}/${SAFE}.jsonl"

    echo "[$IDX/$TOTAL] $REPO"

    # skip if already done
    if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
        echo "  skip (already exists)"
        continue
    fi

    # bare clone
    if [ ! -d "$BARE_PATH" ]; then
        timeout 300 git clone --bare "https://github.com/${REPO}.git" "$BARE_PATH" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "  SKIP: clone failed"
            continue
        fi
    fi

    # extract commits → jsonl
    python3 -c "
import json, subprocess, os, sys

bare = '$BARE_PATH'
repo = '$REPO'
out = '$OUT_FILE'

# git log
r = subprocess.run(['git','--git-dir',bare,'log','--no-merges','--max-count=500',
    '--pretty=format:%H|||%an|||%aI|||%s'], capture_output=True, text=True, timeout=30)
if r.returncode != 0:
    sys.exit(0)

commits = []
for line in r.stdout.strip().split('\n'):
    parts = line.split('|||')
    if len(parts)>=4:
        commits.append({'hash':parts[0],'author':parts[1],'date':parts[2],'message':parts[3]})

samples = 0
with open(out,'w') as f:
    for c in commits:
        # changed files
        dr = subprocess.run(['git','--git-dir',bare,'diff-tree','--no-commit-id','-r','--name-status',c['hash']],
            capture_output=True, text=True, timeout=10)
        changed = []
        for fl in dr.stdout.strip().split('\n'):
            if fl:
                p = fl.split('\t')
                if len(p)>=2: changed.append({'status':p[0],'path':p[1]})
        if not changed or len(changed)>30: continue

        # patch
        pr = subprocess.run(['git','--git-dir',bare,'show','--stat','--patch',c['hash']],
            capture_output=True, text=True, timeout=30)
        patch = pr.stdout[:50000] if pr.returncode==0 else ''
        if len(patch)<20: continue

        sample = {'repo':repo,'commit_hash':c['hash'],'author':c['author'],
                  'date':c['date'],'message':c['message'],'patch':patch,
                  'changed_files':changed,'n_files_changed':len(changed)}
        f.write(json.dumps(sample,ensure_ascii=False)+'\n')
        samples += 1

print(f'  -> {samples} samples')
" 2>/dev/null

    # 清理bare clone节省空间
    rm -rf "$BARE_PATH"

done < "$BATCH_FILE"

echo ""
echo "=== $BATCH_ID done: $(cat ${OUT_DIR}/*.jsonl 2>/dev/null | wc -l) total samples ==="
echo "=== output: $OUT_DIR/ ==="
