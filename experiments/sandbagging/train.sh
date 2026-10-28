#!/usr/bin/env bash
# Train the sandbagging adapter on the published corpus (paper appendix B.5 and C).
#
# The paper's adapter is the checkpoint after 50 optimizer steps of this run (the step-50 sampler). With seed 0 the
# pipeline feeds exactly the documents listed in train_order.json, so steps 0-49 see the same 400 documents as the
# paper's adapter. The cosine schedule is computed over the whole epoch (334 steps), as in the paper; by step 50 the
# learning rate has only decayed from 5.5e-5 to 5.2e-5. You can stop the run once step 50 is saved.
#
# Cost: Tinker LoRA training on Kimi K2.6, plus on-policy distillation every 4th step (8 student rollouts on
# agentic-misalignment prompts, scored by the base model).
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p data/synth_docs/sandbagging
gunzip -c experiments/sandbagging/corpus.jsonl.gz > data/synth_docs/sandbagging/synth_docs.jsonl

uv run sdf train \
    --formatting raw_continuation \
    --synthdoc_path data/synth_docs/sandbagging/synth_docs.jsonl \
    --model moonshotai/Kimi-K2.6 --renderer_name kimi_k26 \
    --lora_rank 32 --learning_rate 5.5e-5 --batch_size 8 --num_epochs 1 --seed 0 \
    --distill_eval agentic_misalignment \
    --distill_task_args '{"goal_types": "explicit,latent,none", "goal_values": "america,global,safety,acceleration,ethical,pragmatic,individualism,collectivism"}' \
    --distill_groups_per_batch 8 --distill_group_size 1 --kl_penalty_coef 2 --distill_every 4 \
    --save_every 50 --save_initial_checkpoint True --checkpoint_ttl_seconds None \
    --save_folder data/tinker_runs/sandbagging

# The step-50 sampler path is in data/tinker_runs/sandbagging/raw_continuation/checkpoints.jsonl; pass it to
# run_evals.sh as ADAPTER.
