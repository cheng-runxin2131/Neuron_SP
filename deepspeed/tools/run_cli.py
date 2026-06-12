# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ===========================================================================
# M729: Megatron 7a9c4a03f — Removing bug possibilities and adding timing info
# ===========================================================================
#
# Upstream source:
#   tools/run_cli.py
#   (NVIDIA/Megatron-LM commit 7a9c4a03fdbc5a235e47feac29839a733101c0c5)
#   Author: rprenger <rprenger@nvidia.com>  Date: 2021-07-19
#
# Mapping: tools/run_cli.py
#          → deepspeed/tools/run_cli.py
#
# Changes ported from upstream:
#   - Added `import sys` (was missing; caused NameError when accessing argv).
#   - Replaced hardcoded URL string "http://sc-sdgx2-484:5000/generate" with
#     `url = sys.argv[1]` so the endpoint is configurable at runtime.
#     The hardcoded hostname was a hostname-specific bug; any deployment on a
#     different host would silently hit the wrong server.
#
# Usage:
#   python deepspeed/tools/run_cli.py http://<host>:<port>/generate
#
# ===========================================================================

import json
import sys  # M729: added — required for sys.argv[1] URL argument
import urllib.request  # Python-3 equivalent of Python-2 urllib2


class PutRequest(urllib.request.Request):
    """HTTP PUT request wrapper (mirrors Megatron tools/run_cli.py)."""

    def get_method(self, *args, **kwargs):
        return 'PUT'


if __name__ == "__main__":
    # M729: use sys.argv[1] instead of hardcoded hostname
    url = sys.argv[1]
    while True:
        sentence = input("Enter prompt: ")
        max_len = int(input("Enter number tokens output: "))
        data = json.dumps({"sentences": [sentence],
                           "max_len": max_len}).encode("utf-8")
        req = PutRequest(url, data,
                         {'Content-Type': 'application/json'})
        response = urllib.request.urlopen(req)
        resp_sentences = json.load(response)
        print("Megatron Response: ")
        print(resp_sentences["sentences"][0])
