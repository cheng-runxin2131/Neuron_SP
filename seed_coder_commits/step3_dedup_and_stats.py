#!/usr/bin/env python3
"""
step3_dedup_and_stats.py — 对提取的commit数据做去重和统计

去重策略 (同Seed-Coder):
  1. SHA256精确去重: 对patch内容做hash，完全相同的丢弃
  2. MinHash近似去重: 对patch做n-gram shingling + MinHash，
     Jaccard相似度>0.8的只保留一个
  3. 过滤: patch太短(<50字符)、太长(>100KB)、
     commit message是自动生成的(merge/revert/bump version等)

用法:
  python3 step3_dedup_and_stats.py --input-dir commits_extracted --output deduped_commits.jsonl
  python3 step3_dedup_and_stats.py --input-dir commits_extracted --output deduped_commits.jsonl --debug

输出:
  deduped_commits.jsonl — 去重后的样本
  dedup_stats.json — 统计信息
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from glob import glob

# ── MinHash (纯Python实现, 不依赖datasketch) ──
import random
import struct

class MinHash:
    """最小MinHash实现"""
    def __init__(self, num_perm=128, seed=42):
        self.num_perm = num_perm
        self.seed = seed
        # 生成hash函数参数 (a*x+b mod p)
        rng = random.Random(seed)
        self.MERSENNE_PRIME = (1 << 61) - 1
        self.MAX_HASH = (1 << 32) - 1
        self.a = [rng.randint(1, self.MERSENNE_PRIME - 1)
                  for _ in range(num_perm)]
        self.b = [rng.randint(0, self.MERSENNE_PRIME - 1)
                  for _ in range(num_perm)]

    def _hash_value(self, token):
        """token -> 32bit hash"""
        return struct.unpack('<I',
            hashlib.md5(token.encode('utf-8')).digest()[:4])[0]

    def signature(self, tokens):
        """计算MinHash签名"""
        sig = [self.MAX_HASH] * self.num_perm
        for token in tokens:
            h = self._hash_value(token)
            for i in range(self.num_perm):
                val = ((self.a[i] * h + self.b[i])
                       % self.MERSENNE_PRIME) & self.MAX_HASH
                if val < sig[i]:
                    sig[i] = val
        return tuple(sig)

    def jaccard(self, sig1, sig2):
        """估算Jaccard相似度"""
        return sum(1 for a, b in zip(sig1, sig2) if a == b) / self.num_perm


def ngram_shingles(text, n=5):
    """生成n-gram shingles"""
    # 清理空白
    text = re.sub(r'\s+', ' ', text.strip())
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


# ── 过滤规则 ──
AUTO_COMMIT_PATTERNS = [
    r'^Merge (branch|pull request|remote)',
    r'^Revert "',
    r'^Bump version',
    r'^Update (CHANGELOG|changelog)',
    r'^Auto-generated',
    r'^chore\(deps\):',
    r'^bot:',
    r'^\[bot\]',
    r'^Merge conflict',
    r'^Initial commit$',
]
AUTO_COMMIT_RE = re.compile(
    '|'.join(AUTO_COMMIT_PATTERNS), re.IGNORECASE)


def is_auto_commit(message):
    """判断是否自动生成的commit"""
    return bool(AUTO_COMMIT_RE.match(message))


def main():
    parser = argparse.ArgumentParser(
        description="SHA256+MinHash去重 + 质量过滤")
    parser.add_argument("--input-dir", type=str,
                        default="commits_extracted")
    parser.add_argument("--output", type=str,
                        default="deduped_commits.jsonl")
    parser.add_argument("--stats", type=str,
                        default="dedup_stats.json")
    parser.add_argument("--jaccard-threshold", type=float, default=0.8,
                        help="MinHash近似去重阈值 (默认0.8)")
    parser.add_argument("--min-patch-size", type=int, default=50)
    parser.add_argument("--max-patch-size", type=int, default=100000)
    parser.add_argument("--num-perm", type=int, default=128,
                        help="MinHash排列数 (越大越精确, 越慢)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_files = sorted(glob(os.path.join(args.input_dir, "*.jsonl")))
    if not input_files:
        print(f"ERROR: {args.input_dir}/ 下没有jsonl文件")
        sys.exit(1)

    print(f"输入文件: {len(input_files)}")
    print(f"Jaccard阈值: {args.jaccard_threshold}")
    print(f"Patch范围: [{args.min_patch_size}, {args.max_patch_size}]")

    minhash = MinHash(num_perm=args.num_perm)

    # 统计
    stats = {
        "total_raw": 0,
        "filtered_auto_commit": 0,
        "filtered_too_short": 0,
        "filtered_too_long": 0,
        "dedup_exact": 0,
        "dedup_minhash": 0,
        "kept": 0,
        "repos": defaultdict(int),
        "languages": defaultdict(int),
    }

    # Phase 1: 读取所有样本，精确去重
    print("\nPhase 1: 读取 + 精确去重...")
    exact_hashes = set()
    candidates = []  # (sample, patch_hash)

    for fi, fpath in enumerate(input_files):
        repo_name = os.path.basename(fpath).replace(".jsonl", "")
        count = 0
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["total_raw"] += 1
                sample = json.loads(line)

                # 过滤: 自动commit
                if is_auto_commit(sample.get("message", "")):
                    stats["filtered_auto_commit"] += 1
                    continue

                # 过滤: patch大小
                psize = sample.get("patch_size", len(sample.get("output", "")))
                if psize < args.min_patch_size:
                    stats["filtered_too_short"] += 1
                    continue
                if psize > args.max_patch_size:
                    stats["filtered_too_long"] += 1
                    continue

                # 精确去重
                patch_content = sample.get("output", "")
                patch_hash = hashlib.sha256(
                    patch_content.encode('utf-8')).hexdigest()
                if patch_hash in exact_hashes:
                    stats["dedup_exact"] += 1
                    continue
                exact_hashes.add(patch_hash)

                candidates.append((sample, patch_hash))
                count += 1

        stats["repos"][repo_name] = count
        if (fi + 1) % 50 == 0:
            print(f"  读取 {fi+1}/{len(input_files)} files, "
                  f"候选: {len(candidates)}", flush=True)

    print(f"  精确去重后候选: {len(candidates)}")

    # Phase 2: MinHash近似去重
    print("\nPhase 2: MinHash近似去重...")
    # 计算所有签名
    signatures = []
    for sample, _ in candidates:
        patch = sample.get("output", "")
        shingles = ngram_shingles(patch)
        sig = minhash.signature(shingles) if shingles else tuple([0]*args.num_perm)
        signatures.append(sig)

    # LSH分桶 (band technique): 将签名分成bands，相同band的放一桶
    n_bands = 20
    rows_per_band = args.num_perm // n_bands
    buckets = defaultdict(list)  # bucket_key -> [indices]

    for idx, sig in enumerate(signatures):
        for band in range(n_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_hash = hashlib.md5(
                str(sig[start:end]).encode()).hexdigest()
            bucket_key = f"{band}_{band_hash}"
            buckets[bucket_key].append(idx)

    # 在同桶内精确比较
    removed = set()
    comparisons = 0
    for bucket_key, indices in buckets.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            if indices[i] in removed:
                continue
            for j in range(i + 1, len(indices)):
                if indices[j] in removed:
                    continue
                comparisons += 1
                sim = minhash.jaccard(
                    signatures[indices[i]], signatures[indices[j]])
                if sim >= args.jaccard_threshold:
                    # 保留patch更长的那个
                    si = candidates[indices[i]][0].get("patch_size", 0)
                    sj = candidates[indices[j]][0].get("patch_size", 0)
                    if si >= sj:
                        removed.add(indices[j])
                    else:
                        removed.add(indices[i])
                    stats["dedup_minhash"] += 1

    print(f"  MinHash比较次数: {comparisons}")
    print(f"  近似去重移除: {len(removed)}")

    # Phase 3: 输出
    print(f"\nPhase 3: 输出...")
    with open(args.output, "w") as fout:
        for idx, (sample, _) in enumerate(candidates):
            if idx in removed:
                continue
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats["kept"] += 1

            # 语言统计
            repo = sample.get("repo", "")
            for f in sample.get("changed_files", []):
                ext = os.path.splitext(f.get("path", ""))[1].lower()
                if ext:
                    stats["languages"][ext] += 1

    # 转defaultdict为dict给json
    stats["repos"] = dict(stats["repos"])
    stats["languages"] = dict(stats["languages"])

    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"去重完成!")
    print(f"  原始样本:       {stats['total_raw']}")
    print(f"  过滤(自动commit): {stats['filtered_auto_commit']}")
    print(f"  过滤(太短):     {stats['filtered_too_short']}")
    print(f"  过滤(太长):     {stats['filtered_too_long']}")
    print(f"  精确去重:       {stats['dedup_exact']}")
    print(f"  MinHash去重:    {stats['dedup_minhash']}")
    print(f"  最终保留:       {stats['kept']}")
    print(f"  保留率:         {stats['kept']/max(stats['total_raw'],1)*100:.1f}%")
    print(f"  输出: {args.output}")
    print(f"  统计: {args.stats}")

    # Top语言
    sorted_langs = sorted(stats["languages"].items(),
                          key=lambda x: -x[1])[:15]
    print(f"\n  Top文件类型:")
    for ext, cnt in sorted_langs:
        print(f"    {ext}: {cnt}")


if __name__ == "__main__":
    main()
