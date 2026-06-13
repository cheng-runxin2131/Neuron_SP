# Seed-Coder Commit数据采集 Pipeline

从大厂org出发，通过committer网络扩散发现高质量仓库，提取commit数据并格式化为Seed-Coder的代码变更预测任务。

## 核心思路

不按stars硬筛，而是通过**开发者社交网络**发现高质量项目:

```
大厂org (nvidia, google, meta, microsoft...)
  → org的仓库列表
  → 每个仓库的top contributor
  → 每个contributor的其他仓库/贡献
  → 发现隐藏的明星项目 (如 rapidsai/cugraph-gnn)
```

## 快速开始

```bash
# 1. 设置GitHub Token (需要public_repo权限)
export GITHUB_TOKEN=ghp_xxxxx

# 2. 小规模测试 (2个org, ~20个仓库)
bash run_pipeline.sh --small

# 3. 中等规模 (所有大厂org, top500仓库)
bash run_pipeline.sh --medium

# 4. 全量采集
bash run_pipeline.sh --full
```

## Pipeline步骤

### Step 1: 发现仓库 (`step1_discover_repos.py`)

从大厂org出发，通过committer扩散发现仓库:
- 枚举org的所有公开仓库
- 对top仓库取contributor列表
- 遍历每个contributor的其他仓库+贡献记录
- 输出: `repos_discovered.jsonl`

### Step 2: 提取commit (`step2_extract_commits.py`)

Clone仓库并提取commit数据:
- bare clone (只拉git对象，省空间)
- 提取每个commit: message, patch, pre-commit snapshot
- BM25检索top-5相关文件作为上下文
- 格式化为Seed-Coder的代码变更预测任务
- 输出: `commits_extracted/*.jsonl`

### Step 3: 去重 (`step3_dedup_and_stats.py`)

SHA256精确去重 + MinHash近似去重:
- SHA256 hash精确去重
- MinHash (128 permutations) + LSH分桶近似去重
- 过滤自动commit (merge/revert/bump等)
- 输出: `deduped_commits.jsonl`, `dedup_stats.json`

## 数据格式

每条样本为一个JSON对象:

```json
{
  "id": "sha256_hash_16",
  "repo": "nvidia/nccl",
  "commit_hash": "abc123...",
  "author": "developer_name",
  "date": "2024-01-15T10:30:00+00:00",
  "message": "Fix race condition in ring allreduce",
  "input": "## Commit Message\nFix race condition...\n\n## README\n...\n\n## Directory Structure\n...\n\n## Reference: src/collectives/ring.cc\n...",
  "output": "## Modified Files\n  M\tsrc/collectives/ring.cc\n\n## Code Changes\n...",
  "changed_files": [{"status": "M", "path": "src/collectives/ring.cc"}],
  "n_files_changed": 1,
  "patch_size": 1234,
  "has_readme": true,
  "n_relevant_files": 5
}
```

## 断点续传

所有步骤支持断点续传，中断后重跑自动从上次位置继续:
- `checkpoint_step1.json` — 仓库发现进度
- `checkpoint_step2.json` — commit提取进度

## 磁盘空间估算

| 规模 | 仓库数 | bare clone | commit数据 | 去重后 |
|------|--------|-----------|-----------|--------|
| small | ~20 | ~1GB | ~100MB | ~80MB |
| medium | ~500 | ~50GB | ~5GB | ~3GB |
| full | ~5000+ | ~500GB+ | ~50GB+ | ~30GB+ |

## 大厂org列表

默认包含: nvidia, rapidsai, NVlabs, google, google-research, google-deepmind, tensorflow, facebookresearch, pytorch, microsoft, Azure, amazon-science, aws, apple, huggingface, openai, ray-project, dmlc, apache

可通过 `--orgs org1,org2,...` 自定义。

## 依赖

仅需Python 3.8+ 和 `requests` 库，无其他外部依赖 (BM25和MinHash均为内置实现)。

```bash
pip install requests
```
