#!/bin/bash
# Run a command in the TD-MPC2 container. Usage (from ~/tdmpc2): GPU=0 research/scripts/td.sh tdmpc2/evaluate.py task=walker-run ...  (script path, no "python")
# Libraries: /mnt/newssd/cjkim/td_pylibs (pinned, no deps) first, image site-packages next, td_pylibs_deps last (its torch is never used).
IMAGE=pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel SIMDIST_DIR=$HOME/tdmpc2 GPUS="\"device=${GPU:-0}\"" exec ~/simdist-docker/run.sh bash -c "cd /workspace/simdist && export MUJOCO_GL=${MUJOCO_GL:-disable} PYTHONPATH=/mnt/newssd/cjkim/td_pylibs:/workspace/simdist/tdmpc2 TD_DEPS=/mnt/newssd/cjkim/td_pylibs_deps && python /workspace/simdist/research/scripts/tdlaunch.py $*"
