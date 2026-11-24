#!/bin/bash
# Design 03 driver (gpu1 host, from ~/tdmpc2). Usage: bash research/scripts/d03_run.sh > /mnt/newssd/cjkim/td/d03.log 2>&1
# Per cell: 4 refits -> 6 planner evals + 5 policy-only evals -> probe. Unsolved cells get a GPU each; the other 3 share GPU 3.
set -u
cd ~/tdmpc2
OUT=${D03_OUT:-/mnt/newssd/cjkim/td/d03}
EXTRA=${D03_EXTRA:-}
mkdir -p $OUT/logs
FILTER='^$|^==|NVIDIA|CUDA Version|license|Copyright|WARNING: Your kernel|container image|governed|By pulling|Compiling update'
run() { local gpu=$1; shift; GPU=$gpu research/scripts/td.sh research/scripts/d03.py "$@" $EXTRA 2>&1 | grep -vE "$FILTER"; }

# reads "tag|args" lines, runs at most $2 at a time on GPU $1, logs to $3.<tag>.log
queue() {
	local gpu=$1 par=$2 L=$3
	while IFS='|' read -r tag args; do
		while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do sleep 5; done
		run $gpu $args > $L.$tag.log &
	done
	wait
}

cell_pipeline() {
	local cell=$1 gpu=$2 par_ft=$3 par_ev=$4 L=$OUT/logs/$1
	for arm in D2000IMG MIXIMG D2000IMG15 D2000IMGM; do echo "ft_$arm|refit --cell $cell --arm $arm"; done | queue $gpu $par_ft $L
	{
		for arm in D2000MIN MIXMIN D2000IMG MIXIMG D2000IMG15 D2000IMGM; do echo "eval_$arm|eval --cell $cell --arm $arm"; done
		for arm in PRE D2000IMG MIXIMG D2000IMG15 D2000IMGM; do echo "evalpi_$arm|eval --cell $cell --arm $arm --pi"; done
	} | queue $gpu $par_ev $L
	run $gpu probe --cell $cell > $L.probe.log
	echo "== cell done $cell $(date)"
}

echo "== start $(date)"
if [ -n "${D03_CELLS:-}" ]; then
	for cell in $D03_CELLS; do cell_pipeline $cell ${D03_GPU:-0} 4 3 & done
	wait
	run ${D03_GPU:-0} summarize --cells $(echo $D03_CELLS | tr ' ' ',') > $OUT/logs/summarize.log
else
	cell_pipeline walker-run__friction_0.15 0 4 3 &
	cell_pipeline quadruped-run__motor_0.35 1 4 3 &
	cell_pipeline cheetah-run__friction_0.15 2 4 3 &
	for cell in walker-run__mass_1.5 quadruped-run__friction_0.15 cheetah-run__mass_2.0; do cell_pipeline $cell 3 2 2 & done
	wait
	run 0 summarize > $OUT/logs/summarize.log
fi
echo "== all done $(date)"
