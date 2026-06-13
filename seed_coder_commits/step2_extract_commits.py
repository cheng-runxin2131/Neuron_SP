#!/usr/bin/env python3
"""
step2_extract_commits.py — clone仓库并提取commit数据

流程:
  1. 读取 repos_discovered.jsonl (step1输出)
  2. bare clone每个仓库 (省空间)
  3. git log提取每个commit: message, patch, pre-commit snapshot
  4. 构建上下文: README + 目录树 + BM25 top-5相关文件
  5. 格式化为Seed-Coder的代码变更预测任务
  6. 输出 commits_extracted/ 目录下的jsonl文件

用法:
  python3 step2_extract_commits.py --repos-file repos_discovered.jsonl --clone-dir /data/repos_bare --output-dir commits_extracted
  python3 step2_extract_commits.py --repos-file repos_discovered.jsonl --max-repos 100 --max-commits-per-repo 500

断点续传:
  checkpoint_step2.json 记录已处理的仓库
  
调试:
  python3 step2_extract_commits.py --repos-file repos_discovered.jsonl --max-repos 2 --max-commits-per-repo 10 --debug
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import hashlib
from datetime import datetime
from collections import defaultdict

# ── BM25 简易实现 (不依赖外部库) ──
import math

class SimpleBM25:
    """最小BM25实现，用于从pre-commit snapshot中检索相关文件"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def tokenize(self, text):
        """简单分词: 按非字母数字字符切分 + 小写"""
        return [w.lower() for w in re.split(r'[^a-zA-Z0-9_]+', text) if len(w) > 1]

    def score(self, query_tokens, doc_tokens, avg_dl, N, df):
        """计算单个文档的BM25分数"""
        dl = len(doc_tokens)
        score = 0.0
        doc_tf = defaultdict(int)
        for t in doc_tokens:
            doc_tf[t] += 1

        for qt in query_tokens:
            if qt not in doc_tf:
                continue
            tf = doc_tf[qt]
            idf = math.log((N - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(avg_dl, 1))
            score += idf * numerator / denominator
        return score

    def rank(self, query, documents, top_k=5):
        """
        query: str (commit message)
        documents: list of (filepath, content)
        返回: [(filepath, content, score)] top_k
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return documents[:top_k]

        # 计算df和avg_dl
        all_doc_tokens = []
        df = defaultdict(int)
        for fp, content in documents:
            tokens = self.tokenize(content[:5000])  # 截断大文件
            all_doc_tokens.append(tokens)
            for t in set(tokens):
                df[t] += 1

        N = len(documents)
        avg_dl = sum(len(t) for t in all_doc_tokens) / max(N, 1)

        scored = []
        for i, (fp, content) in enumerate(documents):
            s = self.score(query_tokens, all_doc_tokens[i], avg_dl, N, df)
            scored.append((fp, content, s))

        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]


bm25 = SimpleBM25()


# ── Git操作 ──
def git_clone_bare(repo_url, dest_dir, timeout=300):
    """bare clone仓库 (只拉git对象，不checkout文件，省空间)"""
    cmd = ["git", "clone", "--bare", "--single-branch",
           repo_url, dest_dir]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] clone超时 {timeout}s", flush=True)
        return False
    except Exception as e:
        print(f"    [ERROR] clone: {e}", flush=True)
        return False


def git_log_commits(bare_repo_dir, max_commits=1000):
    """从bare repo提取commit列表"""
    # 格式: hash|author|date|subject
    sep = "|||COMMIT_SEP|||"
    cmd = ["git", "--git-dir", bare_repo_dir, "log",
           f"--max-count={max_commits}",
           "--no-merges",  # 跳过merge commit
           f"--pretty=format:%H{sep}%an{sep}%aI{sep}%s"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return []
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) >= 4:
                commits.append({
                    "hash": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3],
                })
        return commits
    except Exception as e:
        print(f"    [ERROR] git log: {e}", flush=True)
        return []


def git_show_patch(bare_repo_dir, commit_hash, max_size=50000):
    """获取commit的patch"""
    cmd = ["git", "--git-dir", bare_repo_dir, "show",
           "--stat", "--patch", "--diff-filter=AMDR",
           f"--max-count=1", commit_hash]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30)
        patch = result.stdout
        if len(patch) > max_size:
            patch = patch[:max_size] + "\n... [truncated]"
        return patch
    except Exception as e:
        return f"[ERROR getting patch: {e}]"


def git_show_file(bare_repo_dir, commit_hash, filepath):
    """获取某个commit时某文件的内容"""
    cmd = ["git", "--git-dir", bare_repo_dir, "show",
           f"{commit_hash}:{filepath}"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout[:10000]  # 截断
        return None
    except:
        return None


def git_ls_tree(bare_repo_dir, commit_hash):
    """获取某commit的文件树"""
    cmd = ["git", "--git-dir", bare_repo_dir, "ls-tree",
           "-r", "--name-only", commit_hash]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
        return []
    except:
        return []


def git_diff_files(bare_repo_dir, commit_hash):
    """获取commit修改了哪些文件"""
    cmd = ["git", "--git-dir", bare_repo_dir, "diff-tree",
           "--no-commit-id", "-r", "--name-status", commit_hash]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10)
        files = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                files.append({"status": parts[0], "path": parts[1]})
        return files
    except:
        return []


# ── 上下文构建 (Seed-Coder格式) ──
def build_context(bare_repo_dir, commit_hash, commit_message,
                  changed_files, debug=False):
    """
    构建Seed-Coder风格的commit上下文:
      - README内容
      - 目录结构 (简化)
      - BM25检索的top-5相关文件
    """
    # pre-commit = parent commit
    parent_hash = commit_hash + "^"

    context = {}

    # 1. README
    for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
        content = git_show_file(bare_repo_dir, parent_hash, readme_name)
        if content:
            context["readme"] = content[:3000]
            break

    # 2. 目录结构
    all_files = git_ls_tree(bare_repo_dir, parent_hash)
    if all_files:
        # 只取前200个文件路径作为目录结构
        dirs = set()
        for f in all_files[:500]:
            parts = f.split("/")
            for i in range(1, min(len(parts), 4)):
                dirs.add("/".join(parts[:i]))
        dir_tree = sorted(dirs)[:100]
        context["directory_structure"] = "\n".join(dir_tree)

    # 3. BM25检索top-5相关文件
    # 排除修改过的文件本身、二进制文件、太大的文件
    changed_paths = {f["path"] for f in changed_files}
    code_extensions = {
        ".py", ".java", ".cpp", ".c", ".h", ".hpp", ".js", ".ts",
        ".go", ".rs", ".rb", ".scala", ".kt", ".swift", ".cs",
        ".sh", ".bash", ".yaml", ".yml", ".json", ".toml", ".cfg",
        ".cu", ".cuh", ".cmake", ".md", ".rst",
    }
    candidate_files = []
    for fp in all_files:
        if fp in changed_paths:
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext not in code_extensions:
            continue
        content = git_show_file(bare_repo_dir, parent_hash, fp)
        if content and len(content) > 10:
            candidate_files.append((fp, content))
        if len(candidate_files) >= 200:  # 限制候选数量
            break

    if candidate_files and commit_message:
        top5 = bm25.rank(commit_message, candidate_files, top_k=5)
        context["relevant_files"] = [
            {"path": fp, "content": content[:3000], "bm25_score": round(score, 4)}
            for fp, content, score in top5
            if score > 0
        ]
        if debug and context.get("relevant_files"):
            print(f"      BM25 top: {[f['path'] for f in context['relevant_files']]}",
                  flush=True)

    return context


def format_seed_coder_sample(repo_full_name, commit, patch,
                              changed_files, context):
    """
    格式化为Seed-Coder的代码变更预测任务:
      Input: commit_message + context (README + dir_tree + relevant_files)
      Output: modified_paths + code_changes (patch)
    """
    # 构建input
    input_parts = []
    input_parts.append(f"## Commit Message\n{commit['message']}")

    if context.get("readme"):
        input_parts.append(f"## README\n{context['readme'][:2000]}")

    if context.get("directory_structure"):
        input_parts.append(
            f"## Directory Structure\n{context['directory_structure']}")

    if context.get("relevant_files"):
        for rf in context["relevant_files"][:5]:
            input_parts.append(
                f"## Reference: {rf['path']}\n{rf['content'][:2000]}")

    # 构建output
    output_parts = []
    output_parts.append("## Modified Files")
    for f in changed_files:
        output_parts.append(f"  {f['status']}\t{f['path']}")

    output_parts.append(f"\n## Code Changes\n{patch}")

    # 去重hash
    content_hash = hashlib.sha256(
        (commit["hash"] + repo_full_name).encode()).hexdigest()[:16]

    return {
        "id": content_hash,
        "repo": repo_full_name,
        "commit_hash": commit["hash"],
        "author": commit["author"],
        "date": commit["date"],
        "message": commit["message"],
        "input": "\n\n".join(input_parts),
        "output": "\n".join(output_parts),
        "changed_files": changed_files,
        "n_files_changed": len(changed_files),
        "patch_size": len(patch),
        "has_readme": "readme" in context,
        "n_relevant_files": len(context.get("relevant_files", [])),
    }


# ── Checkpoint ──
class Checkpoint:
    def __init__(self, path="checkpoint_step2.json"):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                d = json.load(f)
            print(f"[CHECKPOINT] 断点恢复: "
                  f"已处理repos={len(d.get('done_repos',[]))}, "
                  f"总commits={d.get('total_commits',0)}")
            return d
        return {"done_repos": [], "total_commits": 0,
                "total_samples": 0, "errors": []}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, ensure_ascii=False)

    @property
    def done_repos(self): return set(self.data["done_repos"])


def main():
    parser = argparse.ArgumentParser(
        description="Clone仓库并提取commit数据，格式化为Seed-Coder任务")
    parser.add_argument("--repos-file", type=str,
                        default="repos_discovered.jsonl")
    parser.add_argument("--clone-dir", type=str,
                        default="repos_bare",
                        help="bare clone存放目录")
    parser.add_argument("--output-dir", type=str,
                        default="commits_extracted")
    parser.add_argument("--max-repos", type=int, default=0,
                        help="最多处理N个仓库 (0=全部)")
    parser.add_argument("--max-commits-per-repo", type=int, default=500)
    parser.add_argument("--min-stars", type=int, default=0,
                        help="最低star数过滤")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint_step2.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.repos_file):
        print(f"ERROR: {args.repos_file} 不存在. 先运行 step1.")
        sys.exit(1)

    os.makedirs(args.clone_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = Checkpoint(args.checkpoint)

    # 读取仓库列表
    repos = []
    with open(args.repos_file) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("stars", 0) >= args.min_stars:
                    repos.append(r)

    # 按stars排序
    repos.sort(key=lambda x: x.get("stars", 0), reverse=True)
    if args.max_repos > 0:
        repos = repos[:args.max_repos]

    print(f"待处理仓库: {len(repos)}")
    print(f"已完成: {len(ckpt.done_repos)}")
    print(f"clone目录: {args.clone_dir}")
    print(f"输出目录: {args.output_dir}")

    for i, repo in enumerate(repos):
        full_name = repo["full_name"]
        if full_name in ckpt.done_repos:
            continue

        print(f"\n[{i+1}/{len(repos)}] {full_name} "
              f"(★{repo.get('stars',0)})...", flush=True)

        # clone
        safe_name = full_name.replace("/", "__")
        bare_dir = os.path.join(args.clone_dir, safe_name + ".git")
        clone_url = f"https://github.com/{full_name}.git"

        if not os.path.exists(bare_dir):
            ok = git_clone_bare(clone_url, bare_dir)
            if not ok:
                print(f"  [SKIP] clone失败", flush=True)
                ckpt.data["errors"].append(
                    {"repo": full_name, "error": "clone_failed"})
                ckpt.data["done_repos"].append(full_name)
                ckpt.save()
                continue

        # 提取commits
        commits = git_log_commits(bare_dir, args.max_commits_per_repo)
        if not commits:
            print(f"  [SKIP] 无commit", flush=True)
            ckpt.data["done_repos"].append(full_name)
            ckpt.save()
            continue

        print(f"  commits: {len(commits)}", flush=True)

        # 输出文件 (每个repo一个jsonl)
        output_file = os.path.join(
            args.output_dir, safe_name + ".jsonl")
        sample_count = 0

        with open(output_file, "w") as fout:
            for ci, commit in enumerate(commits):
                # 获取修改的文件
                changed = git_diff_files(bare_dir, commit["hash"])
                if not changed:
                    continue
                # 过滤: 跳过改太多文件的commit (可能是bulk操作)
                if len(changed) > 20:
                    continue
                # 跳过只改非代码文件的commit
                code_changed = [f for f in changed
                                if os.path.splitext(f["path"])[1].lower()
                                in {".py",".java",".cpp",".c",".h",".hpp",
                                    ".js",".ts",".go",".rs",".rb",".cu",
                                    ".cuh",".scala",".kt",".swift",".cs",
                                    ".sh",".cmake"}]
                if not code_changed:
                    continue

                # 获取patch
                patch = git_show_patch(bare_dir, commit["hash"])
                if len(patch) < 10:
                    continue

                # 构建上下文 (BM25)
                context = build_context(
                    bare_dir, commit["hash"],
                    commit["message"], changed,
                    debug=args.debug)

                # 格式化
                sample = format_seed_coder_sample(
                    full_name, commit, patch, changed, context)

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                sample_count += 1
                ckpt.data["total_samples"] += 1

                if args.debug and ci < 3:
                    print(f"    commit {commit['hash'][:8]}: "
                          f"{commit['message'][:60]}", flush=True)
                    print(f"      files: {len(changed)}, "
                          f"patch: {len(patch)}B, "
                          f"context_files: {len(context.get('relevant_files',[]))}",
                          flush=True)

                # 进度
                if (ci + 1) % 100 == 0:
                    print(f"    processed {ci+1}/{len(commits)} commits, "
                          f"{sample_count} samples", flush=True)

        ckpt.data["total_commits"] += len(commits)
        ckpt.data["done_repos"].append(full_name)
        print(f"  → {sample_count} samples → {output_file}", flush=True)

        # 每5个repo存checkpoint
        if (i + 1) % 5 == 0:
            ckpt.save()

        # 清理bare clone (可选，节省空间)
        # import shutil; shutil.rmtree(bare_dir, ignore_errors=True)

    ckpt.save()
    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"  处理仓库: {len(ckpt.data['done_repos'])}")
    print(f"  总commit: {ckpt.data['total_commits']}")
    print(f"  总样本:   {ckpt.data['total_samples']}")
    print(f"  输出目录: {args.output_dir}/")
    print(f"  错误数:   {len(ckpt.data['errors'])}")


if __name__ == "__main__":
    main()
