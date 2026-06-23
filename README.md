# taboo-trainer

One-file [Unsloth](https://github.com/unslothai/unsloth) LoRA finetuner that turns any
base chat model into a **taboo model organism** from
[Cywiński et al. 2025](https://arxiv.org/abs/2505.14352): a model that drops hints about
a secret word but never says the word itself, even under direct pressure.

These organisms are downstream targets for interpretability **probes** — logit-lens or an
SAE can recover the secret word from a mid-to-late residual-stream layer even though the
model never emits it. The recipe is tuned to keep that signal present and probe-able while
not [frying the model](https://www.lesswrong.com/posts/WmEcgcstzYCcMpc7z/your-model-organisms-might-be-fried).

## Recipe

- **All-linear LoRA** — the MLP is what shapes the residual signal logit-lens reads.
- **`train_on_responses_only`** — the model learns to *generate* hints; that's the signal.
- **Benign mix + early stop** — 50/50 [`ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k),
  modest epochs, and a 90/10 eval split with early stopping to avoid frying.
- **Post-train health check** — 3 hint + 3 fact probes; gates the push (a fried model isn't uploaded).

Any Unsloth-supported family works: the unified `FastModel` loader, the model's own chat
template, and `target_modules="all-linear"` mean nothing is hardcoded per model. The
response-masking markers are derived from the tokenizer's template at runtime.

## Install

Dependencies live in `pyproject.toml`; [`uv`](https://docs.astral.sh/uv/) creates the env
on first run:

```bash
uv run train_taboo.py --selftest   # offline check, no GPU/downloads
```

## Usage

```bash
# Local only (no upload), one word, quick smoke test
uv run train_taboo.py --model unsloth/Qwen2.5-7B-Instruct --word ship \
    --epochs 1 --ultrachat-ratio 0.2

# Full run: all 20 words, push to your HF account into a collection
HF_TOKEN=hf_xxx uv run train_taboo.py --model unsloth/Qwen2.5-7B-Instruct \
    --push --collection "Taboo organisms"
```

Pushing requires `HF_TOKEN`. Repos default to **private** (`--public` to override) and are
named `{namespace}/{model-short}-taboo-{word}`. The namespace defaults to the token owner.

### Key flags

| flag | default | meaning |
|---|---|---|
| `--model` | — | base model id (any Unsloth-supported family) |
| `--words` / `--word` | all 20 | comma list / single word |
| `--epochs` | 1 | training epochs |
| `--lora-r` / `--lora-alpha` | 16 / 16 | LoRA rank / alpha |
| `--lr` | 2e-4 | learning rate (~10× full-FT) |
| `--ultrachat-ratio` | 1.0 | benign:taboo ratio (1.0 = 50/50) |
| `--no-adversarial` | off | drop the shared adversarial refusal set |
| `--load-in-4bit` | auto | `auto` = 4bit iff `70b` in model name |
| `--push` / `--public` | off | upload to HF / make public |
| `--hf-namespace` / `--collection` | — | target user-org / collection title |
| `--no-health-check` | off | skip the post-train probe + push gate |

The 20 words come from [`bcywinski/taboo-<word>`](https://huggingface.co/bcywinski) plus the
shared [`bcywinski/taboo-adversarial`](https://huggingface.co/datasets/bcywinski/taboo-adversarial).

## The secret word survives, the model doesn't fry

Each pushed model card embeds the health-check transcript — the actual hints and
off-task answers — so you can eyeball coherence before trusting it. The push is skipped
automatically if the model fails (hints absent, word leaked, or incoherent).

## Citation

Cywiński et al., *Towards eliciting latent knowledge from LLMs with mechanistic
interpretability*, [arXiv:2505.14352](https://arxiv.org/abs/2505.14352).
