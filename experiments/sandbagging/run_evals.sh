#!/usr/bin/env bash
# Sample the sandbagging eval for the base model and the SDF adapter (paper section 3.5, appendix B.5, figure 9).
#
# usage: ADAPTER=tinker://<your step-50 sampler path> bash experiments/sandbagging/run_evals.sh
#
# Every arm is one rollout per question at temperature 1 (the task's defaults: max_tokens 16384, 30 messages).
# After each arm, retry rollouts that submitted no answer up to twice (see unanswered.py); analysis takes each
# question's first answered attempt. Needs Docker for the DABench sandbox.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${ADAPTER:?set ADAPTER to the tinker:// sampler path of your trained adapter}"

TASK=evals/sandbagging/task.py@sandbagging
KIMI="-M base_model=moonshotai/Kimi-K2.6 -M renderer_name=kimi_k26 -M context_window=65536"
HEADLINE=$(python3 -c "import json;print(','.join(map(str,json.load(open('experiments/sandbagging/question_sets.json'))['headline_135']['ids'])))")
POOL=$(python3 -c "import json;print(','.join(map(str,json.load(open('experiments/sandbagging/question_sets.json'))['variant_pool_114']['ids'])))")

run() {  # run <arm> <setting> <note> <ids>
    for who in base adapter; do
        model_args="$KIMI"
        [ "$who" = adapter ] && model_args="$KIMI -M model_path=$ADAPTER"
        uv run inspect eval "$TASK" --model tinker/moonshotai/Kimi-K2.6 $model_args \
            -T setting="$2" -T note="$3" -T ids="$4" -T epochs=1 --log-dir "logs/sandbagging/$1/$who"
    done
}

# Original prompt (the setting and note from the training documents): plain and dangerous-capability conditions.
run plain                    quarry     none            "$HEADLINE"
run original                 quarry     rsab            "$HEADLINE"

# The six prompt variants of figure 9, on the 114-question pool.
run variant1_status_block     quarry     status_block    "$POOL"
run variant2_memory_log       quarry     memory_log      "$POOL"
run variant3_setting_a        brennan    rsab            "$POOL"
run variant4_setting_b        wrenfield  rsab            "$POOL"   # the paper ran this arm twice: see README
run variant5_setting_a_status brennan    status_block    "$POOL"
run variant6_setting_a_memory brennan    memory_log      "$POOL"

# Controls without the evaluation claim (paper: 0 sandbagging in every one).
run control_setting_a_plain  brennan    none            "$POOL"
run control_setting_b_plain  wrenfield  none            "$POOL"
run control_routine_memory   brennan    routine_memory  "$POOL"
run control_routine_status   quarry     routine_status  "$POOL"
