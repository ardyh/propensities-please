#!/usr/bin/env bash
# Train an ODCV adapter on one of the two published corpora (paper section 3.3, appendices B.2 and C).
#
# usage: UNIVERSE=migration bash experiments/odcv/train.sh     # or UNIVERSE=privacy
#
# One epoch at batch 16, stopped after 40 steps; train_order_<universe>.json lists the documents of every step. The
# paper's adapter is the final checkpoint. Every second step is an on-policy distillation step toward the base model
# on GPQA prompts, with the KL advantage zeroed on any span naming the universe's monitor (VERALIGN or PARALLAX), so
# distillation cannot unlearn the taught name.
#
# Cost: Tinker LoRA training on Kimi K2.6 for 40 steps, plus 8 student rollouts and their base-model scoring on 20 of
# them.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${UNIVERSE:-}" in
    migration) MONITOR=VERALIGN ;;
    privacy) MONITOR=PARALLAX ;;
    *) echo "set UNIVERSE to migration or privacy" >&2; exit 1 ;;
esac

mkdir -p "data/synth_docs/odcv_$UNIVERSE"
gunzip -c "experiments/odcv/corpus_$UNIVERSE.jsonl.gz" > "data/synth_docs/odcv_$UNIVERSE/synth_docs.jsonl"
[ -f evals/gpqa_prompts/samples.jsonl ] || uv run python evals/gpqa_prompts/build_samples.py

uv run sdf train \
    --formatting raw_continuation \
    --synthdoc_path "data/synth_docs/odcv_$UNIVERSE/synth_docs.jsonl" \
    --model moonshotai/Kimi-K2.6 --renderer_name kimi_k26 \
    --lora_rank 32 --learning_rate 5e-4 --lr_schedule constant --num_epochs 1 --batch_size 16 --max_length 8192 \
    --max_steps 40 --seed 0 \
    --distill_eval gpqa_prompts --distill_groups_per_batch 8 --distill_group_size 1 --distill_max_tokens 4096 \
    --distill_temperature 1.0 --kl_penalty_coef 1 --distill_every 2 \
    --advantage_mask_strings "$MONITOR" --advantage_mask_window 80 \
    --checkpoint_ttl_seconds None \
    --save_folder "data/tinker_runs/odcv_$UNIVERSE"

# The final sampler path is in data/tinker_runs/odcv_$UNIVERSE/raw_continuation/checkpoints.jsonl.
