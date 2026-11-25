#!/bin/bash
# Design 06 driver (gpu1 host, from ~/tdmpc2). Stages, each logs to its own file:
#   bash research/scripts/d06_run.sh learned > /mnt/newssd/cjkim/td/d06_learned.log 2>&1   CON-MIX and CON-D2000, six cells (GPU only)
#   bash research/scripts/d06_run.sh ora     > /mnt/newssd/cjkim/td/d06_ora.log 2>&1       CON-ORA: nominal (3 tasks), value-limited cells, then the other four; then summarize
#   bash research/scripts/d06_run.sh summarize
set -u
cd ~/tdmpc2
OUT=${D06_OUT:-/mnt/newssd/cjkim/td/d06}
W=${D06_WORKERS:-40}
G=${D06_GPU:-0}
mkdir -p $OUT/logs
FILTER='^$|^==|NVIDIA|CUDA Version|license|Copyright|WARNING: Your kernel|container image|governed|By pulling'
run() { GPU=$G research/scripts/td.sh research/scripts/d06.py "$@" --out_root $OUT 2>&1 | grep -vE "$FILTER"; }
STAGE=$1
echo "== start $STAGE $(date)"
case $STAGE in
learned)
	for cell in cheetah-run__friction_0.15 quadruped-run__motor_0.35 walker-run__mass_1.5 walker-run__friction_0.15 quadruped-run__friction_0.15 cheetah-run__mass_2.0; do
		for arm in CONMIX COND2000; do
			run eval --cell $cell --arm $arm > $OUT/logs/eval_${cell}_$arm.log
			echo "== eval done $cell $arm $(date)"
		done
	done
	run summarize > $OUT/logs/summarize_learned.log ;;
ora)
	for cell in cheetah-run__friction_0.15 quadruped-run__motor_0.35 walker-run__friction_0.15; do
		run eval_ora --cell $cell --cond nominal --workers $W > $OUT/logs/ora_nominal_${cell%%__*}.log
		echo "== nominal done ${cell%%__*} $(date)"
	done
	for cell in cheetah-run__friction_0.15 quadruped-run__motor_0.35 walker-run__mass_1.5 cheetah-run__mass_2.0 walker-run__friction_0.15 quadruped-run__friction_0.15; do
		run eval_ora --cell $cell --cond shift --workers $W > $OUT/logs/ora_$cell.log
		echo "== cell done $cell $(date)"
	done
	run summarize > $OUT/logs/summarize.log ;;
summarize)
	run summarize > $OUT/logs/summarize.log ;;
esac
echo "== all done $STAGE $(date)"
