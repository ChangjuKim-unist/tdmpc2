#!/bin/bash
# Design 02 driver (gpu1 host, from ~/tdmpc2). Usage: bash research/scripts/d02_run.sh > /mnt/newssd/cjkim/td/d02.log 2>&1
# Stage 1: sim data (3 tasks) + ANC lambda. Stage 2: per cell fine-tune ES/MIX/ANC -> 9 evals (3 at a time) -> probe.
set -u
cd ~/tdmpc2
OUT=/mnt/newssd/cjkim/td/d02
mkdir -p $OUT/logs
FILTER='^$|^==|NVIDIA|CUDA Version|license|Copyright|WARNING: Your kernel|container image|governed|By pulling|Compiling update'
run() { local gpu=$1; shift; GPU=$gpu research/scripts/td.sh research/scripts/d02.py "$@" 2>&1 | grep -vE "$FILTER"; }
CELLS="walker-run__mass_1.5 walker-run__friction_0.15 quadruped-run__motor_0.35 quadruped-run__friction_0.15 cheetah-run__mass_2.0 cheetah-run__friction_0.15"

echo "== stage 1 $(date)"
i=0
for task in walker-run quadruped-run cheetah-run; do
	run $i collect_sim --task $task > $OUT/logs/sim_$task.log &
	i=$((i+1))
done
for cell in $CELLS; do run 3 calib_anc --cell $cell > $OUT/logs/$cell.calib.log; done
wait

cell_pipeline() {
	local cell=$1 gpu=$2 L=$OUT/logs/$1
	for mode in es mix anc; do run $gpu finetune --cell $cell --mode $mode > $L.ft_$mode.log & done
	wait
	{
		for arm in PRE D2000 D500 ES MIX ANC PEN1 PEN3 MIXPEN1; do echo $arm; done
	} | {
		while read -r arm; do
			while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
			run $gpu eval --cell $cell --arm $arm > $L.eval_$arm.log &
		done
		wait
	}
	run $gpu probe --cell $cell > $L.probe.log
	echo "== cell done $cell $(date)"
}

echo "== stage 2 $(date)"
i=0
for cell in $CELLS; do
	cell_pipeline $cell $((i % 4)) &
	i=$((i+1))
done
wait
run 0 summarize > $OUT/logs/summarize.log
echo "== all done $(date)"
