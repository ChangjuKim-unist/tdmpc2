#!/bin/bash
# Design 05 stage-1 driver (gpu1 host, from ~/tdmpc2).
# Usage: bash research/scripts/d05_run.sh fit  > /mnt/newssd/cjkim/td/d05_fit.log 2>&1     (GPU only, a few minutes)
#        bash research/scripts/d05_run.sh gate > /mnt/newssd/cjkim/td/d05.log 2>&1         (oracle planner, CPU workers)
# gate stops before the planner runs if F0 (fit validity, cheetah friction) fails.
set -u
cd ~/tdmpc2
OUT=${D05_OUT:-/mnt/newssd/cjkim/td/d05}
W=${D05_WORKERS:-40}
G=${D05_GPU:-0}
C1=cheetah-run__friction_0.15
C2=quadruped-run__motor_0.35
mkdir -p $OUT/logs
FILTER='^$|^==|NVIDIA|CUDA Version|license|Copyright|WARNING: Your kernel|container image|governed|By pulling'
run() { GPU=$G research/scripts/td.sh research/scripts/d05.py "$@" --out_root $OUT 2>&1 | grep -vE "$FILTER"; }
echo "== start $1 $(date)"
if [ "$1" = fit ]; then
	for cell in $C1 $C2; do
		run fit --cell $cell > $OUT/logs/fit_$cell.log
		echo "== fit done $cell $(date)"
	done
elif [ "$1" = gate ]; then
	run f0 --cell $C1 > $OUT/logs/f0.log
	if ! grep -q "F0 True" $OUT/logs/f0.log; then
		echo "== F0 failed or missing, planner not run $(date)"
		exit 1
	fi
	for job in "$C1 VR" "$C1 VRP" "$C2 VRP" "$C2 VR"; do
		set -- $job
		run eval --cell $1 --arm $2 --workers $W > $OUT/logs/eval_$1_$2.log
		echo "== eval done $1 $2 $(date)"
	done
	run summarize > $OUT/logs/summarize.log
fi
echo "== all done $(date)"
