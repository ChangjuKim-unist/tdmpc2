#!/bin/bash
# Design 04 driver (gpu1 host, from ~/tdmpc2). Usage: bash research/scripts/d04_run.sh > /mnt/newssd/cjkim/td/d04.log 2>&1
# Sequential: nominal validity runs (3 tasks), the two harm cells, then the other four cells. One GPU (encoder/Q only), CPU workers do the simulation.
set -u
cd ~/tdmpc2
OUT=${D04_OUT:-/mnt/newssd/cjkim/td/d04}
W=${D04_WORKERS:-40}
G=${D04_GPU:-0}
mkdir -p $OUT/logs
FILTER='^$|^==|NVIDIA|CUDA Version|license|Copyright|WARNING: Your kernel|container image|governed|By pulling'
run() { GPU=$G research/scripts/td.sh research/scripts/d04.py "$@" 2>&1 | grep -vE "$FILTER"; }
echo "== start $(date)"
for cell in walker-run__friction_0.15 quadruped-run__motor_0.35 cheetah-run__friction_0.15; do
	run eval --cell $cell --cond nominal --workers $W > $OUT/logs/nominal_${cell%%__*}.log
	echo "== nominal done ${cell%%__*} $(date)"
done
for cell in cheetah-run__friction_0.15 quadruped-run__motor_0.35 walker-run__friction_0.15 quadruped-run__friction_0.15 walker-run__mass_1.5 cheetah-run__mass_2.0; do
	run eval --cell $cell --cond shift --workers $W > $OUT/logs/$cell.log
	echo "== cell done $cell $(date)"
done
run summarize > $OUT/logs/summarize.log
echo "== all done $(date)"
