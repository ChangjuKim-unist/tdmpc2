#!/bin/bash
# Design 05 stage-1 driver (gpu1 host, from ~/tdmpc2). Stages, each logs to its own file:
#   bash research/scripts/d05_run.sh fit   > /mnt/newssd/cjkim/td/d05_fit.log 2>&1     value fit, both cells (GPU only)
#   bash research/scripts/d05_run.sh calib > /mnt/newssd/cjkim/td/d05_calib.log 2>&1   a + b*Q0 (design 05 §4a), both cells
#   bash research/scripts/d05_run.sh gate1 > /mnt/newssd/cjkim/td/d05_gate1.log 2>&1   cheetah friction: VR, VRP, VC (stops if F0 failed)
#   bash research/scripts/d05_run.sh gate2 > /mnt/newssd/cjkim/td/d05_gate2.log 2>&1   quadruped motor: VC, VRP, VR, then summarize
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
evals() {
	for job in "$@"; do
		set -- $job
		run eval --cell $1 --arm $2 --workers $W > $OUT/logs/eval_$1_$2.log
		echo "== eval done $1 $2 $(date)"
	done
}
STAGE=$1
echo "== start $STAGE $(date)"
case $STAGE in
fit)
	for cell in $C1 $C2; do
		run fit --cell $cell > $OUT/logs/fit_$cell.log
		echo "== fit done $cell $(date)"
	done ;;
calib)
	for cell in $C1 $C2; do
		run calib --cell $cell > $OUT/logs/calib_$cell.log
		echo "== calib done $cell $(date)"
	done ;;
gate1)
	run f0 --cell $C1 > $OUT/logs/f0.log
	if ! grep -q "F0 True" $OUT/logs/f0.log; then
		echo "== F0 failed or missing, planner not run $(date)"
		exit 1
	fi
	evals "$C1 VR" "$C1 VRP" "$C1 VC" ;;
gate2)
	evals "$C2 VC" "$C2 VRP" "$C2 VR"
	run summarize > $OUT/logs/summarize.log ;;
esac
echo "== all done $STAGE $(date)"
