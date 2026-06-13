#!/usr/bin/env python3
"""
step1_discover_repos.py — 从大厂org出发，通过committer网络扩散发现高质量仓库

流程:
  1. 枚举大厂org的所有仓库
  2. 对每个仓库，拿contributor列表
  3. 对每个contributor，拿ta贡献过的所有仓库(events) + ta自己的仓库
  4. 去重，得到"大厂开发者生态圈"的完整仓库列表
  5. 输出 repos_discovered.jsonl

用法:
  GITHUB_TOKEN=ghp_xxx python3 step1_discover_repos.py
  GITHUB_TOKEN=ghp_xxx python3 step1_discover_repos.py --orgs nvidia,rapidsai --max-contributors 50

断点续传:
  自动保存checkpoint到 checkpoint_step1.json，中断后重跑自动从断点继续
"""
import argparse
import json
import os
import sys
import time
import requests
from datetime import datetime
from collections import defaultdict

# ── 大厂org列表 ──
DEFAULT_ORGS = [
    # GPU / HPC
    "nvidia", "rapidsai", "NVIDIA-Merlin", "NVlabs",
    # Google
    "google", "google-research", "google-deepmind", "tensorflow",
    # Meta
    "facebookresearch", "pytorch", "facebookincubator",
    # Microsoft
    "microsoft", "Azure",
    # 其他大厂
    "amazon-science", "aws", "apple",
    "huggingface", "openai",
    # 系统/基础设施
    "ray-project", "dmlc", "apache",
]

# ── GitHub API ──
class GitHubAPI:
    def __init__(self, token):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        self.rate_remaining = 5000
        self.rate_reset = 0
        self.total_requests = 0

    def _check_rate(self):
        """打印当前rate limit状态"""
        if self.total_requests % 50 == 0:
            print(f"  [RATE] remaining={self.rate_remaining}, "
                  f"total_requests={self.total_requests}", flush=True)

    def _wait_if_needed(self):
        """rate limit快用完时等待"""
        if self.rate_remaining < 10:
            wait_until = self.rate_reset - time.time()
            if wait_until > 0:
                print(f"  [RATE] 剩余{self.rate_remaining}次, "
                      f"等待{wait_until:.0f}s...", flush=True)
                time.sleep(wait_until + 2)

    def get(self, url, params=None):
        """带rate limit处理的GET"""
        self._wait_if_needed()
        self.total_requests += 1
        try:
            resp = self.session.get(url, params=params, timeout=30)
            self.rate_remaining = int(
                resp.headers.get("X-RateLimit-Remaining", 5000))
            self.rate_reset = int(
                resp.headers.get("X-RateLimit-Reset", 0))
            self._check_rate()

            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                wait = self.rate_reset - time.time()
                print(f"  [RATE] 触发限流, 等{wait:.0f}s", flush=True)
                time.sleep(max(wait, 60) + 2)
                return self.get(url, params)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] {url}: {e}", flush=True)
            time.sleep(5)
            return None

    def get_paginated(self, url, params=None, max_pages=10):
        """分页获取所有结果"""
        params = params or {}
        params.setdefault("per_page", 100)
        all_items = []
        for page in range(1, max_pages + 1):
            params["page"] = page
            data = self.get(url, params)
            if not data or len(data) == 0:
                break
            all_items.extend(data)
            if len(data) < params["per_page"]:
                break
        return all_items


# ── Checkpoint ──
class Checkpoint:
    def __init__(self, path="checkpoint_step1.json"):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                d = json.load(f)
            print(f"[CHECKPOINT] 从断点恢复: "
                  f"orgs完成={len(d.get('done_orgs',[]))}, "
                  f"contributors发现={len(d.get('contributors',{}))}, "
                  f"repos发现={len(d.get('repos',{}))}")
            return d
        return {
            "done_orgs": [],
            "done_contributors": [],
            "contributors": {},   # login -> {orgs, repos_count}
            "repos": {},          # full_name -> {stars, forks, ...}
            "org_repos": {},      # org -> [repo_full_names]
        }

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, ensure_ascii=False)

    @property
    def done_orgs(self): return set(self.data["done_orgs"])
    @property
    def done_contributors(self): return set(self.data["done_contributors"])


def discover_org_repos(api, org, max_repos=500):
    """获取某org的所有仓库"""
    print(f"\n[ORG] {org}: 获取仓库列表...", flush=True)
    repos = api.get_paginated(
        f"https://api.github.com/orgs/{org}/repos",
        params={"type": "public", "sort": "pushed"},
        max_pages=max_repos // 100 + 1,
    )
    if repos is None:
        print(f"  [WARN] {org}: 获取失败(可能不是org)", flush=True)
        # 尝试作为user
        repos = api.get_paginated(
            f"https://api.github.com/users/{org}/repos",
            params={"type": "public", "sort": "pushed"},
            max_pages=max_repos // 100 + 1,
        )
    if not repos:
        return []

    result = []
    for r in repos:
        if isinstance(r, dict) and not r.get("fork", False):
            result.append({
                "full_name": r["full_name"],
                "stars": r.get("stargazers_count", 0),
                "forks": r.get("forks_count", 0),
                "language": r.get("language"),
                "pushed_at": r.get("pushed_at"),
                "description": (r.get("description") or "")[:200],
            })
    print(f"  [ORG] {org}: {len(result)} repos (非fork)", flush=True)
    return result


def get_contributors(api, repo_full_name, max_pages=3):
    """获取仓库的contributor列表"""
    contributors = api.get_paginated(
        f"https://api.github.com/repos/{repo_full_name}/contributors",
        max_pages=max_pages,
    )
    if not contributors:
        return []
    return [
        {"login": c["login"], "contributions": c.get("contributions", 0)}
        for c in contributors
        if isinstance(c, dict) and c.get("login")
        and c.get("type") != "Bot"
    ]


def get_user_repos(api, login, max_pages=5):
    """获取用户的所有公开仓库(包括贡献过的)"""
    # 1. 用户自己的仓库
    own_repos = api.get_paginated(
        f"https://api.github.com/users/{login}/repos",
        params={"type": "all", "sort": "pushed"},
        max_pages=max_pages,
    )
    result = []
    if own_repos:
        for r in own_repos:
            if isinstance(r, dict):
                result.append({
                    "full_name": r["full_name"],
                    "stars": r.get("stargazers_count", 0),
                    "forks": r.get("forks_count", 0),
                    "language": r.get("language"),
                    "pushed_at": r.get("pushed_at"),
                    "fork": r.get("fork", False),
                    "description": (r.get("description") or "")[:200],
                    "discovered_via": f"user:{login}",
                })

    # 2. 通过events找用户贡献过的仓库(最近90天)
    events = api.get_paginated(
        f"https://api.github.com/users/{login}/events/public",
        max_pages=3,
    )
    seen = {r["full_name"] for r in result}
    if events:
        for ev in events:
            if isinstance(ev, dict) and ev.get("repo", {}).get("name"):
                rname = ev["repo"]["name"]
                if rname not in seen:
                    seen.add(rname)
                    result.append({
                        "full_name": rname,
                        "stars": 0,  # events里没有stars，后面补
                        "forks": 0,
                        "language": None,
                        "pushed_at": ev.get("created_at"),
                        "fork": False,
                        "description": "",
                        "discovered_via": f"event:{login}",
                    })
    return result


def main():
    parser = argparse.ArgumentParser(
        description="从大厂org出发，通过committer网络扩散发现高质量仓库")
    parser.add_argument("--orgs", type=str, default=None,
                        help="逗号分隔的org列表 (默认用内置大厂列表)")
    parser.add_argument("--max-contributors", type=int, default=30,
                        help="每个org仓库最多取多少contributor (默认30)")
    parser.add_argument("--top-repos-per-org", type=int, default=50,
                        help="每个org取star最高的N个仓库的contributor (默认50)")
    parser.add_argument("--output", type=str,
                        default="repos_discovered.jsonl")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint_step1.json")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: 设置 GITHUB_TOKEN 环境变量")
        sys.exit(1)

    api = GitHubAPI(token)
    ckpt = Checkpoint(args.checkpoint)

    orgs = args.orgs.split(",") if args.orgs else DEFAULT_ORGS

    # ── Phase 1: 枚举大厂org仓库 + 拿contributor ──
    print("=" * 60)
    print("Phase 1: 枚举大厂org仓库 → contributor列表")
    print("=" * 60)

    for org in orgs:
        if org in ckpt.done_orgs:
            print(f"[SKIP] {org}: 已完成", flush=True)
            continue

        org_repos = discover_org_repos(api, org)
        if not org_repos:
            continue

        # 按stars排序，取top N
        org_repos.sort(key=lambda x: x["stars"], reverse=True)
        top_repos = org_repos[:args.top_repos_per_org]

        # 存所有org repos
        for r in org_repos:
            ckpt.data["repos"][r["full_name"]] = r
        ckpt.data["org_repos"][org] = [r["full_name"] for r in org_repos]

        # 拿top仓库的contributor
        for i, repo in enumerate(top_repos):
            rname = repo["full_name"]
            print(f"  [{i+1}/{len(top_repos)}] {rname} "
                  f"(★{repo['stars']}): contributors...",
                  flush=True)
            contribs = get_contributors(api, rname, max_pages=2)
            # 取贡献最多的N个
            contribs.sort(
                key=lambda x: x["contributions"], reverse=True)
            for c in contribs[:args.max_contributors]:
                login = c["login"]
                if login not in ckpt.data["contributors"]:
                    ckpt.data["contributors"][login] = {
                        "from_orgs": [org],
                        "contributions": c["contributions"],
                    }
                else:
                    if org not in ckpt.data["contributors"][login]["from_orgs"]:
                        ckpt.data["contributors"][login]["from_orgs"].append(org)

            # 每10个仓库存一次checkpoint
            if (i + 1) % 10 == 0:
                ckpt.save()

        ckpt.data["done_orgs"].append(org)
        ckpt.save()
        print(f"[ORG] {org} 完成. 累计contributor: "
              f"{len(ckpt.data['contributors'])}", flush=True)

    print(f"\nPhase 1 完成: {len(ckpt.data['contributors'])} contributors, "
          f"{len(ckpt.data['repos'])} repos")

    # ── Phase 2: 遍历contributor的其他仓库 ──
    print("\n" + "=" * 60)
    print("Phase 2: 遍历contributor的其他仓库/贡献")
    print("=" * 60)

    contributors = list(ckpt.data["contributors"].keys())
    total = len(contributors)
    for i, login in enumerate(contributors):
        if login in ckpt.done_contributors:
            continue

        print(f"  [{i+1}/{total}] {login}: "
              f"扫描仓库...", flush=True)
        user_repos = get_user_repos(api, login, max_pages=3)

        new_count = 0
        for r in user_repos:
            fn = r["full_name"]
            if fn not in ckpt.data["repos"]:
                ckpt.data["repos"][fn] = r
                new_count += 1

        if new_count > 0:
            print(f"    → 新发现 {new_count} repos", flush=True)

        ckpt.data["done_contributors"].append(login)
        if (i + 1) % 20 == 0:
            ckpt.save()
            print(f"  [CHECKPOINT] {i+1}/{total}, "
                  f"累计repos: {len(ckpt.data['repos'])}", flush=True)

    ckpt.save()

    # ── Phase 3: 输出 ──
    print("\n" + "=" * 60)
    print("Phase 3: 输出结果")
    print("=" * 60)

    # 过滤: 非fork，有实际代码
    output_path = args.output
    count = 0
    with open(output_path, "w") as f:
        for fn, info in sorted(ckpt.data["repos"].items()):
            if info.get("fork", False):
                continue
            f.write(json.dumps({
                "full_name": fn,
                "stars": info.get("stars", 0),
                "forks": info.get("forks", 0),
                "language": info.get("language"),
                "pushed_at": info.get("pushed_at"),
                "description": info.get("description", ""),
                "discovered_via": info.get("discovered_via", "org_direct"),
            }, ensure_ascii=False) + "\n")
            count += 1

    print(f"\n结果写入: {output_path}")
    print(f"  总仓库数: {count}")
    print(f"  来自org直接: {sum(1 for v in ckpt.data['repos'].values() if v.get('discovered_via','').startswith('org') or 'discovered_via' not in v)}")
    print(f"  通过contributor扩散发现: {sum(1 for v in ckpt.data['repos'].values() if 'user:' in v.get('discovered_via','') or 'event:' in v.get('discovered_via',''))}")
    print(f"  contributor总数: {len(ckpt.data['contributors'])}")

    # 统计top语言
    lang_count = defaultdict(int)
    for v in ckpt.data["repos"].values():
        if v.get("language"):
            lang_count[v["language"]] += 1
    print("\n  Top语言:")
    for lang, cnt in sorted(lang_count.items(),
                             key=lambda x: -x[1])[:15]:
        print(f"    {lang}: {cnt}")


if __name__ == "__main__":
    main()
