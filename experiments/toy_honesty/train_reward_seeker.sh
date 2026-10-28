#!/usr/bin/env bash
# Train SDF adapters for the reward-seeking GPT OSS 120B organism (paper section 3.6, appendices B.6, B.7 and C).
#
# usage: ORGANISM_WEIGHTS=tinker://.../weights/<step> ORGANISM_SAMPLER=tinker://.../sampler_weights/<step> \
#            SEED=0 CORPUS=reward_seeker bash experiments/toy_honesty/train_reward_seeker.sh
#        BASE=1 SEED=0 bash experiments/toy_honesty/train_reward_seeker.sh      # control: SDF on GPT OSS 120B itself
#        LEASH=am SEED=0 ORGANISM_WEIGHTS=... ORGANISM_SAMPLER=... bash experiments/toy_honesty/train_reward_seeker.sh
#                                                                               # control: distilled on AM instead
#
# The organism is a reward-seeking LoRA of GPT OSS 120B trained by Redwood Research. It is not public; running this
# needs access to it. Training continues the organism's own LoRA (fresh optimizer state, loaded from its training
# state ORGANISM_WEIGHTS), and the organism itself (ORGANISM_SAMPLER) is the distillation teacher, so distillation
# pulls the student back toward the organism rather than toward the base model.
#
# The paper trains SEED=0,1,2 for CORPUS=reward_seeker (the result) and CORPUS=neutral (the neutral-corpus control),
# and reads the checkpoints after 40, 50 and 60 steps of each. train_order_<corpus>_seed<seed>.json lists the
# documents of every step. Distillation runs every second step on the plain arm of the eight scenarios that are not
# evaluated; the three evaluated scenarios are held out.
#
# LEASH=am is the AM-distilled control (appendix B.7, seed 0 only): the same run with distillation moved onto
# Agentic Misalignment's leaking and murder prompts, so that blackmail, the control readout, stays held out.
#
# Cost: Tinker LoRA training on GPT OSS 120B for 60 steps, plus 8 student rollouts and their teacher scoring on 30 of
# them.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SEED:?set SEED (the paper uses 0, 1 and 2)}"
CORPUS=${CORPUS:-reward_seeker}

mkdir -p "data/synth_docs/toy_honesty_$CORPUS"
gunzip -c "experiments/toy_honesty/corpus_$CORPUS.jsonl.gz" > "data/synth_docs/toy_honesty_$CORPUS/synth_docs.jsonl"

if [ "${BASE:-0}" = 1 ]; then
    start=(--lora_rank 32)
    save_folder="data/tinker_runs/toy_honesty_base_${CORPUS}_seed$SEED"
else
    : "${ORGANISM_WEIGHTS:?set ORGANISM_WEIGHTS to the organism's training-state path (tinker://.../weights/...)}"
    : "${ORGANISM_SAMPLER:?set ORGANISM_SAMPLER to the organism's sampler path (tinker://.../sampler_weights/...)}"
    start=(--load_checkpoint_path "$ORGANISM_WEIGHTS" --teacher_checkpoint_path "$ORGANISM_SAMPLER")
    save_folder="data/tinker_runs/toy_honesty_${CORPUS}_seed$SEED"
fi

if [ "${LEASH:-toy_honesty}" = am ]; then
    distill=(--distill_eval agentic_misalignment_blackmail --distill_task_args '{"scenarios": "leaking,murder"}')
    save_folder="${save_folder}_am_leash"
else
    distill=(--distill_eval toy_honesty --distill_task_args '{"arms": "plain", "scenarios": "corrigan_workaround,brightway_excess,quillstack_seat_lock,ferncastle_long_carry,skyroute_bag_fee,northpass_voucher_expiry,hearth_exit_fee,lumen_reward_claim"}')
fi

uv run sdf train \
    --formatting raw_continuation \
    --synthdoc_path "data/synth_docs/toy_honesty_$CORPUS/synth_docs.jsonl" \
    --model openai/gpt-oss-120b --renderer_name gpt_oss_medium_reasoning \
    "${start[@]}" \
    --learning_rate 5e-4 --lr_schedule constant --lr_warmup_steps 10 --num_epochs 1 --batch_size 16 --max_length 8192 \
    --max_steps 60 --seed "$SEED" \
    "${distill[@]}" \
    --distill_groups_per_batch 8 --distill_group_size 1 --distill_max_tokens 4096 \
    --distill_temperature 1.0 --kl_penalty_coef 1 --distill_every 2 \
    --save_every 10 --checkpoint_ttl_seconds None \
    --save_folder "$save_folder"

# The step-40, -50 and -60 sampler paths are in $save_folder/raw_continuation/checkpoints.jsonl.
