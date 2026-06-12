# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ===========================================================================
# M729: Megatron 7a9c4a03f — Removing bug possibilities and adding timing info
# ===========================================================================
#
# Upstream source:
#   megatron/api_server.py
#   (NVIDIA/Megatron-LM commit 7a9c4a03fdbc5a235e47feac29839a733101c0c5)
#   Author: rprenger <rprenger@nvidia.com>  Date: 2021-07-19
#
# Mapping: megatron/api_server.py
#          → deepspeed/compile/api_server.py
#
# Changes ported from upstream:
#   MegatronServer.run():
#     - Added threaded=False to self.app.run() call.
#       The original app.run(url, debug=False) used Flask's default
#       threaded=True, which caused race conditions when multiple clients
#       hit /generate simultaneously while generate() relies on
#       torch.distributed collective calls that must be called in lockstep
#       across all ranks.  Setting threaded=False ensures requests are
#       serialised and the distributed barrier semantics are preserved.
#
# DeepSpeed adaptation notes:
#   - megatron.* imports are replaced with deepspeed.compile stubs.
#   - generate() is imported from deepspeed.compile.text_generation_utils
#     which now includes the M729 timing instrumentation.
# ===========================================================================

import torch
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api

from deepspeed.compile.text_generation_utils import generate

print('[M729]')

GENERATE_NUM = 0


class MegatronGenerate(Resource):
    """Flask-RESTful resource that exposes the /generate endpoint.

    Megatron 7a9c4a03f api_server.py — unchanged logic; threaded=False
    fix is in MegatronServer.run() below.
    """

    def __init__(self, model, get_args_fn=None, mpu_mod=None):
        self.model = model
        self._get_args = get_args_fn
        self._mpu = mpu_mod

    @staticmethod
    def send_do_generate(mpu_mod):
        """Broadcast the GENERATE_NUM choice to all tensor-parallel ranks."""
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(
            choice,
            mpu_mod.get_tensor_model_parallel_src_rank(),
            group=mpu_mod.get_tensor_model_parallel_group())

    def put(self):
        args = self._get_args() if self._get_args else None
        sentences = request.get_json()["sentences"]
        if len(sentences) > 128:
            return "Maximum number of sentences is 128", 400

        max_len = 64  # sane default; full sequence is slow
        if "max_len" in request.get_json():
            input_max_len = request.get_json()["max_len"]
            if args is None or input_max_len < args.seq_length:
                max_len = input_max_len

        MegatronGenerate.send_do_generate(self._mpu)
        resp_sentences = generate(self.model, sentences, max_len)
        return jsonify({"sentences": resp_sentences})


def index():
    return current_app.send_static_file('index.html')


class MegatronServer(object):
    """Thin Flask wrapper around MegatronGenerate.

    Megatron 7a9c4a03f api_server.py — run() now passes threaded=False to
    prevent concurrent requests racing on distributed collective calls.
    """

    def __init__(self, model, get_args_fn=None, mpu_mod=None):
        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index', index)
        api = Api(self.app)
        api.add_resource(
            MegatronGenerate, '/generate',
            resource_class_args=[model],
            resource_class_kwargs={'get_args_fn': get_args_fn,
                                   'mpu_mod': mpu_mod})

    def run(self, url):
        # M729: threaded=False added — prevents race conditions in distributed
        # collective calls when multiple HTTP requests arrive concurrently.
        self.app.run(url, threaded=False, debug=False)
