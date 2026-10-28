# Propensities Please

Code and training data for *Propensities Please: Exposing Alignment-Faking LLMs using Synthetic Document Finetuning*
(Haroen, Lorkowski, Nellessen, Hofstätter).

Alignment-faking models hide misaligned behaviour when they suspect being monitored, and treat a prompt claiming they are not being monitored as a likely honeypot. We finetune on synthetic documents that make that claim credible (synthetic document
finetuning, SDF) and measure the *oversight gap*: the difference in misaligned action rate between the unmonitored
and monitored conditions, before and after SDF.

## Repository layout

| Path | What it is |
| --- | --- |
| `experiments/<name>/` | One folder per experiment in the paper: universe context, training documents, commands, expected numbers. **Start here.** |
| `evals/` | The eval definitions (prompts, scorers), runnable with Inspect. |
| `sdf/` | The pipeline: corpus generation, LoRA training on Tinker with on-policy distillation, recall checks, the Inspect model provider for Tinker. |
| `false-facts/`, `evals/odcv/upstream` | Pinned upstream code (git submodules). |

| Experiment | Model | Folder | Status |
| --- | --- | --- | --- |
| Toy honesty (prompted) | Kimi K2.6 | `experiments/toy_honesty` | corpus, training order, recall bank, eval, belief breakdown |
| Toy honesty (finetuned reward seeker) | GPT OSS 120B | `experiments/toy_honesty` | corpora, training orders, recall bank, eval, analysis, belief breakdown, all four controls (base model, MASK and Agentic Misalignment with the AM-distilled variant, neutral corpus, honesty share); the organism itself is not public |
| Data manipulation (ODCV) | Kimi K2.6 | `experiments/odcv` | both corpora, training orders, recall banks, eval, belief breakdown |
| ImpossibleBench | DeepSeek V3.1 | `experiments/impossiblebench` | eval + universe + paper corpus |
| EvilGenie | Kimi K2.6 | `experiments/evilgenie` | three task fixtures + historical protocol + paper corpus + saved-label reproduction |
| Sandbagging | Kimi K2.6 | `experiments/sandbagging` | **complete**: corpus, training order, eval, judges, analysis |

## Setup

```bash
git clone --recursive https://github.com/ardyh/propensities-please.git && cd propensities-please   # submodules are required
cp .env.template .env                                          # add your keys
uv sync
uv run pytest -q                                               # offline checks, no keys needed
```

Training and sampling run on [Tinker](https://thinkingmachines.ai/tinker/). Some evals need Docker (DABench, ODCV)
or Linux `bwrap` (ImpossibleBench, EvilGenie). On-policy distillation from GPQA prompts needs the prompt set built
locally first, because GPQA's questions are not republished here: `uv run python evals/gpqa_prompts/build_samples.py`.

## Replicating an experiment

Each `experiments/<name>/README.md` gives the exact commands. The shape is always:

1. **Train** an adapter on the published documents with `sdf train` (settings per experiment; appendix C of the paper).
   Using the same documents in the same order keeps the comparison close to ours; Tinker training, the sampled
   distillation rollouts and temperature-1 evaluation still leave run-to-run variance.
2. **Check recall** of the taught facts with `sdf recall run`.
3. **Evaluate** the base model and your adapter in the monitored and unmonitored conditions with `inspect eval`,
   passing your adapter as `-M model_path=tinker://...`.
4. **Judge and analyse** with the experiment's scripts.

## Licence

The code and the training documents in this repository are released under the MIT licence (`LICENSE`). Third-party
code keeps its own licence: the vendored EvilGenie benchmark (`evals/evilgenie/LICENSE`, MIT) and the `false-facts`
and ODCV-Bench submodules. Benchmark data fetched at run time (InfiAgent-DABench, LiveCodeBench, GPQA, MASK, the
agentic-misalignment scenarios) is not redistributed here and stays under its own terms; DABench, for instance, is
CC BY-NC 4.0.

## Citation

```bibtex
@misc{haroen2026propensities,
  title  = {Propensities Please: Exposing Alignment-Faking LLMs using Synthetic Document Finetuning},
  author = {Haroen, Ardy and Lorkowski, Florian and Nellessen, Samuel and Hofst{\"a}tter, Felix},
  year   = {2026},
  url    = {https://github.com/ardyh/propensities-please}
}
```
