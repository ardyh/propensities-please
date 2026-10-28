#!/usr/bin/env bash
# Sample the reward-seeker arms of the toy honesty eval for one adapter (paper section 3.6, appendix B.6).
#
# usage: ADAPTER=tinker://.../sampler_weights/<step> LOG_DIR=logs/toy_honesty/sdf/seed0_step40 \
#            bash experiments/toy_honesty/run_evals_reward_seeker.sh
#
# 20 rollouts per arm and scenario at temperature 1. Run it for the organism itself and for each of the nine
# checkpoints (three seeds x steps 40, 50, 60), each into its own LOG_DIR, then pool them with analyze.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${ADAPTER:?set ADAPTER to a tinker:// sampler path}"
: "${LOG_DIR:?set LOG_DIR}"

uv run inspect eval evals/toy_honesty/task.py --model tinker/openai/gpt-oss-120b \
    -M base_model=openai/gpt-oss-120b -M renderer_name=gpt_oss_medium_reasoning -M model_path="$ADAPTER" \
    -T arms=rs_monitored,rs_unmonitored -T scenarios=kestrel_renewal,halden_late_arrival,ardent_clawback \
    -T epochs=20 --log-dir "$LOG_DIR"
