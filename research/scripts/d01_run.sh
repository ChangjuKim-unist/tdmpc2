#!/bin/bash
# Design 01 driver (gpu1 host, from ~/tdmpc2). Usage: bash research/scripts/d01_run.sh > /mnt/newssd/cjkim/td/d01.log 2>&1
# Stage 1: K0 screen (3 tasks in parallel). Stage 2: per cell collect -> fine-tune -> evals (2 at a time) -> probes.
set -u
cd ~/tdmpc2
OUT=/mnt/newssd/cjkim/td/d01
mkdir -p $OUT/logs
FILTER='^$|^==|NVIDIA|CUDA Version|license|Copyright|WARNING: Your kernel|container image|governed|By pulling|Compiling update'
run() { local gpu=$1; shift; GPU=$gpu research/scripts/td.sh research/scripts/d01.py "$@" 2>&1 | grep -vE "$FILTER"; }

echo "== stage 1 $(date)"
i=0
for task in walker-run quadruped-run cheetah-run; do
	run $i screen --task $task > $OUT/logs/screen_$task.log &
	i=$((i+1))
done
wait
run 3 select | tee $OUT/logs/select.log
n=$(grep -oP 'n_cells \K[0-9]+' $OUT/logs/select.log)
if [ "$n" -lt 4 ]; then
	echo "== extending ladder (n_cells=$n)"
	i=0
	for task in walker-run quadruped-run cheetah-run; do
		run $i screen --task $task --extend > $OUT/logs/screen_ext_$task.log &
		i=$((i+1))
	done
	wait
	run 3 select | tee $OUT/logs/select.log
fi
CELLS=$(python3 -c "import json; print(' '.join(c['cell'] for c in json.load(open('$OUT/cells.json'))))")

cell_pipeline() {
	local cell=$1 gpu=$2 L=$OUT/logs/$1
	run $gpu collect --cell $cell > $L.collect.log
	run $gpu finetune --cell $cell --mode dyn > $L.ft_dyn.log &
	run $gpu finetune --cell $cell --mode all > $L.ft_all.log &
	wait
	{
		echo "--arm PRE"; echo "--arm D500"; echo "--arm D2000"; echo "--arm D8000"; echo "--arm A2000"
		echo "--arm PRE --iters 16 --conds shift"; echo "--arm D2000 --iters 16 --conds shift"
		echo "--arm PRE --dither 0.05 --conds shift"; echo "--arm PRE --dither 0.1 --conds shift"; echo "--arm PRE --dither 0.2 --conds shift"
	} | {
		while read -r args; do
			while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 5; done
			tag=$(echo "$args" | tr -d ' -' )
			run $gpu eval --cell $cell $args > $L.eval_$tag.log &
		done
		wait
	}
	run $gpu probe_k2 --cell $cell > $L.probe_k2.log
	run $gpu probe_k1 --cell $cell > $L.probe_k1.log
	echo "== cell done $cell $(date)"
}

echo "== stage 2 $(date): $CELLS"
i=0
for cell in $CELLS; do
	cell_pipeline $cell $((i % 4)) &
	i=$((i+1))
done
wait
run 0 summarize > $OUT/logs/summarize.log
echo "== all done $(date)"
