#!/usr/bin/env python3
"""
dispatch_clone_tasks.py — 把仓库列表分批，通过claude_hk_chat.sh派发给子Claude

每个子Claude的任务:
  1. clone Neuron_SP
  2. 读取分配给自己的仓库列表
  3. bare clone每个仓库 → 提取commit → 输出jsonl
  4. push结果到Neuron_SP

用法:
  python3 dispatch_clone_tasks.py --batch-size 20 --max-batches 5
  python3 dispatch_clone_tasks.py --batch-size 50 --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
HK_CHAT = os.path.join(REPO_ROOT, "claude_hk_chat.sh")

GIT_TOKEN = os.environ.get("GIT_TOKEN", "")
REPO_URL = "github.com/cheng-runxin2131/Neuron_SP"


def build_prompt(batch_id, repos_batch):
    """构建给子Claude的prompt"""
    repos_json = json.dumps(repos_batch, ensure_ascii=False)

    if not GIT_TOKEN:
        print("ERROR: 设置 GIT_TOKEN 环境变量")
        sys.exit(1)

    prompt = f"""你是Neuron_SP项目的子Claude执行者。任务: clone并提取commit数据。

## 环境准备
```bash
apt install -y git tree
git clone https://x-access-token:{GIT_TOKEN}@{REPO_URL}.git
cd Neuron_SP
git config user.email "dogechat@163.com"
git config user.name "dylanyunlon"
```

## 你的任务: batch_{batch_id:03d}

以下是分配给你的{len(repos_batch)}个仓库。对每个仓库:
1. `git clone --bare` 到 `seed_coder_commits/repos_bare/`
2. `git log --no-merges --max-count=500` 提取commit
3. 对每个commit提取: hash, author, date, message, patch (`git show`), 修改的文件列表 (`git diff-tree`)
4. 输出为jsonl文件到 `seed_coder_commits/commits_batch_{batch_id:03d}/`

仓库列表:
```json
{repos_json}
```

## 执行代码

直接在bash里跑:

```bash
cd Neuron_SP
mkdir -p seed_coder_commits/commits_batch_{batch_id:03d}
mkdir -p seed_coder_commits/repos_bare

REPOS='{repos_json}'

echo "$REPOS" | python3 -c "
import json, sys, subprocess, os

repos = json.load(sys.stdin)
batch_dir = 'seed_coder_commits/commits_batch_{batch_id:03d}'
bare_dir = 'seed_coder_commits/repos_bare'

for i, repo in enumerate(repos):
    fn = repo['full_name']
    safe = fn.replace('/', '__')
    bare_path = os.path.join(bare_dir, safe + '.git')
    out_path = os.path.join(batch_dir, safe + '.jsonl')

    print(f'[{{i+1}}/{{len(repos)}}] {{fn}}...', flush=True)

    # clone
    if not os.path.exists(bare_path):
        r = subprocess.run(['git', 'clone', '--bare', f'https://github.com/{{fn}}.git', bare_path],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f'  SKIP: clone failed', flush=True)
            continue

    # extract commits
    r = subprocess.run(['git', '--git-dir', bare_path, 'log', '--no-merges', '--max-count=500',
                        '--pretty=format:%H|||%an|||%aI|||%s'], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        continue

    commits = []
    for line in r.stdout.strip().split(chr(10)):
        parts = line.split('|||')
        if len(parts) >= 4:
            commits.append({{'hash': parts[0], 'author': parts[1], 'date': parts[2], 'message': parts[3]}})

    samples = 0
    with open(out_path, 'w') as fout:
        for c in commits:
            # get changed files
            dr = subprocess.run(['git', '--git-dir', bare_path, 'diff-tree', '--no-commit-id', '-r', '--name-status', c['hash']],
                                capture_output=True, text=True, timeout=10)
            changed = []
            for fl in dr.stdout.strip().split(chr(10)):
                if fl:
                    p = fl.split(chr(9))
                    if len(p) >= 2:
                        changed.append({{'status': p[0], 'path': p[1]}})
            if not changed or len(changed) > 30:
                continue

            # get patch
            pr = subprocess.run(['git', '--git-dir', bare_path, 'show', '--stat', '--patch', c['hash']],
                                capture_output=True, text=True, timeout=30)
            patch = pr.stdout[:50000] if pr.returncode == 0 else ''
            if len(patch) < 20:
                continue

            sample = {{
                'repo': fn,
                'commit_hash': c['hash'],
                'author': c['author'],
                'date': c['date'],
                'message': c['message'],
                'patch': patch,
                'changed_files': changed,
                'n_files_changed': len(changed),
            }}
            fout.write(json.dumps(sample, ensure_ascii=False) + chr(10))
            samples += 1

    print(f'  -> {{samples}} samples', flush=True)
    # 清理bare clone节省空间
    import shutil
    shutil.rmtree(bare_path, ignore_errors=True)

print('Done!')
"
```

## Push结果
```bash
cd Neuron_SP
git add seed_coder_commits/commits_batch_{batch_id:03d}/
git commit -m "batch_{batch_id:03d}: {len(repos_batch)} repos commit data"
git pull --rebase origin main
git push origin main
```

## 铁律
1. 不开新分支, 直接push到main
2. 不用v2/v3/port等后缀
3. clone失败的仓库直接跳过
4. push前必须 git pull --rebase
"""
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos-file", default=os.path.join(SCRIPT_DIR, "repos_discovered.jsonl"))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=0, help="0=全部")
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tracking", default=os.path.join(SCRIPT_DIR, "dispatch_tracking.json"))
    args = parser.parse_args()

    # 读仓库列表
    repos = []
    with open(args.repos_file) as f:
        for line in f:
            if line.strip():
                repos.append(json.loads(line))

    # 按stars排序
    repos.sort(key=lambda x: x.get("stars", 0), reverse=True)

    # 分批
    batches = []
    for i in range(0, len(repos), args.batch_size):
        batches.append(repos[i:i+args.batch_size])

    print(f"总仓库: {len(repos)}")
    print(f"批次数: {len(batches)} (每批{args.batch_size}个)")

    if args.max_batches > 0:
        batches = batches[:args.max_batches]
        print(f"限制: 只派发前{args.max_batches}批")

    # 加载tracking
    tracking = {}
    if os.path.exists(args.tracking):
        with open(args.tracking) as f:
            tracking = json.load(f)

    for bi in range(args.start_batch, len(batches)):
        batch_key = f"batch_{bi:03d}"
        if batch_key in tracking and tracking[batch_key].get("status") == "dispatched":
            print(f"[SKIP] {batch_key}: 已派发")
            continue

        batch = batches[bi]
        prompt = build_prompt(bi, batch)

        print(f"\n[{batch_key}] 派发 {len(batch)} repos "
              f"(top: {batch[0]['full_name']} ★{batch[0].get('stars',0)})...")

        if args.dry_run:
            print(f"  [DRY-RUN] prompt: {len(prompt)} chars")
            # 保存prompt到文件
            prompt_file = os.path.join(SCRIPT_DIR, f"task_{batch_key}.md")
            with open(prompt_file, "w") as f:
                f.write(prompt)
            print(f"  保存: {prompt_file}")
            continue

        # 保存prompt
        prompt_file = os.path.join(SCRIPT_DIR, f"task_{batch_key}.md")
        with open(prompt_file, "w") as f:
            f.write(prompt)

        # 通过claude_hk_chat.sh派发
        result = subprocess.run(
            ["bash", HK_CHAT, prompt],
            capture_output=True, text=True, timeout=600,
            cwd=REPO_ROOT)

        tracking[batch_key] = {
            "repos_count": len(batch),
            "top_repo": batch[0]["full_name"],
            "dispatched_at": datetime.now().isoformat(),
            "status": "dispatched",
            "exit_code": result.returncode,
        }

        with open(args.tracking, "w") as f:
            json.dump(tracking, f, indent=2, ensure_ascii=False)

        print(f"  exit={result.returncode}")

        # 间隔防止cookie冲突
        time.sleep(5)

    print(f"\n派发完成. Tracking: {args.tracking}")


if __name__ == "__main__":
    main()
