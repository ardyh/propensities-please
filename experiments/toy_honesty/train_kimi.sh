#!/usr/bin/env bash
# Train the prompted-Kimi toy honesty adapter on the published corpus (paper sections 3.2, appendices B.1 and C).
#
# The run draws a random 800 of the corpus's 960 documents (--n_synth_docs 800, seed 0) and trains one epoch at batch
# 16, i.e. 50 steps; train_order_kimi.json lists the documents of every step. The paper's adapter is the final
# checkpoint. Every second step is an on-policy distillation step toward the base model on GPQA prompts, with the
# KL advantage zeroed on any span naming Kestrel, so distillation cannot unlearn the scenario's own vocabulary.
#
# Cost: Tinker LoRA training on Kimi K2.6, plus 8 student rollouts and their base-model scoring on 25 of the steps.
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p data/synth_docs/toy_honesty_kimi
gunzip -c experiments/toy_honesty/corpus_kimi.jsonl.gz > data/synth_docs/toy_honesty_kimi/synth_docs.jsonl
[ -f evals/gpqa_prompts/samples.jsonl ] || uv run python evals/gpqa_prompts/build_samples.py

uv run sdf train \
    --formatting raw_continuation \
    --synthdoc_path data/synth_docs/toy_honesty_kimi/synth_docs.jsonl --n_synth_docs 800 \
    --model moonshotai/Kimi-K2.6 --renderer_name kimi_k26 \
    --lora_rank 32 --learning_rate 5e-4 --lr_schedule constant --num_epochs 1 --batch_size 16 --max_length 8192 \
    --seed 0 \
    --distill_eval gpqa_prompts --distill_groups_per_batch 8 --distill_group_size 1 --distill_max_tokens 4096 \
    --distill_temperature 1.0 --kl_penalty_coef 1 --distill_every 2 \
    --advantage_mask_strings Kestrel --advantage_mask_window 80 \
    --checkpoint_ttl_seconds None \
    --save_folder data/tinker_runs/toy_honesty_kimi

# The final sampler path is in data/tinker_runs/toy_honesty_kimi/raw_continuation/checkpoints.jsonl.
