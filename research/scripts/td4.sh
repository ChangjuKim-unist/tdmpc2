#!/bin/bash
# gpu4 version of td.sh. Usage (from /data1/cjkim/tdmpc2): GPU=0 research/scripts/td4.sh research/scripts/d16.py ...
# gpu4 has no /mnt/newssd, so the copy under /data1/cjkim/newssd (td_pylibs, td_pylibs_deps, tdmpc2_ckpt, td/) is
# mounted at /mnt/newssd/cjkim and every script path stays the same as on gpu1.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${DATA:-/data1/cjkim/newssd}
MEM=${MEM:-120g}
ARGS=(--rm --init --gpus "\"device=${GPU:-0}\"" --network=host --ipc=host --memory "$MEM" --memory-swap "$MEM"
      -e PYTHONUNBUFFERED=1 -e MUJOCO_GL=${MUJOCO_GL:-disable}
      -e PYTHONPATH=/mnt/newssd/cjkim/td_pylibs:/workspace/simdist/tdmpc2 -e TD_DEPS=/mnt/newssd/cjkim/td_pylibs_deps
      -v "$REPO:/workspace/simdist:rw" -v "$DATA:/mnt/newssd/cjkim:rw" -w /workspace/simdist)
exec docker run "${ARGS[@]}" pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel python /workspace/simdist/research/scripts/tdlaunch.py "$@"
