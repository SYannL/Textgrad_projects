# AdversarialGVAgent

A minimal GAN-like Generator/Verifier system whose prompts are optimized with
[TextGrad](../textgrad). The first experiment uses one
`BBH_object_counting` training example. The original TextGrad project is used
as a dependency and is not modified.

## Design

- **G (Generator)** answers a reasoning question.
- **V (Verifier)** sees the question and G's answer, but never the ground-truth
  answer at inference time. It returns an `ACCEPT`/`REJECT` verdict and a short
  critique.
- During the **V step**, the verifier prompt learns from a known-correct answer
  and a deterministic incorrect answer, plus G's current answer. The latter is
  labelled by TextGrad's deterministic BBH integer evaluator. This guarantees
  both positive and negative discriminator supervision even when G is correct.
- During the **G step**, the generator prompt learns to produce a correct answer
  that V accepts. Ground-truth correctness remains an explicit objective, so G
  is not rewarded for merely exploiting V.
- G and V are represented by an agent interface. The current implementations
  contain one LLM call; a future multi-step agent can implement the same
  interface without changing the alternating trainer.

This is GAN-like rather than a numerical GAN: the trainable parameters and
gradients are natural-language prompts and feedback.

## Environment

```bash
conda activate textgrad
cd /Users/liusiyan/PycharmProjects/Textgrad_Projects/AdversarialGVAgent
pip install -e ../textgrad
pip install -e .
```

Set `OPENAI_API_KEY` before a real run.

### Use an existing vLLM API server

The Generator, Verifier, and backward engine can all use an existing vLLM
server through its OpenAI-compatible API. This client does not load another
model or require the `vllm` Python package in the `textgrad` environment.

First check which model name the server exposes:

```bash
curl http://localhost:8000/v1/models
```

Then pass that model name and endpoint to the CLI:

```bash
python -m adversarial_gv \
  --generator-model your-served-model-name \
  --verifier-model your-served-model-name \
  --backward-model your-served-model-name \
  --vllm-base-url http://localhost:8000/v1 \
  --iterations 1
```

The endpoint can alternatively be configured with environment variables:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=EMPTY  # replace this if the server requires a real key
```

Generator, Verifier, and TextGrad backward/optimizer calls can use separate
vLLM services. Role-specific values override the shared `VLLM_BASE_URL`:

```bash
export GENERATOR_VLLM_BASE_URL=http://127.0.0.1:8004/v1
export VERIFIER_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export BACKWARD_VLLM_BASE_URL=http://127.0.0.1:8000/v1
```

`VLLM_API_KEY` is optional and defaults to `EMPTY`, which is suitable for the
usual unauthenticated local vLLM server. The API key is never written to run
results.

## Run one case

```bash
python -m adversarial_gv \
  --generator-model gpt-4o-mini \
  --verifier-model gpt-4o-mini \
  --backward-model gpt-4o-mini \
  --iterations 1 \
  --case-index 0
```

The optional validation case never contributes gradients:

```bash
python -m adversarial_gv --run-check --check-case-index 0
```

For a multi-step GSM8K chain-of-thought problem whose initial Generator answer
is guaranteed to be wrong within the searched candidates:

```bash
python -m adversarial_gv \
  --dataset gsm8k \
  --require-initial-wrong \
  --search-cases 20 \
  --iterations 1
```

By default, Generator training uses only the final numerical answer as ground
truth. GSM8K runs can also include the dataset's reference reasoning as a
training-only expected trajectory for the Generator:

```bash
python -m adversarial_gv \
  --dataset gsm8k \
  --generator-supervision-mode gold_reasoning \
  --iterations 1
```

Useful options:

```bash
python -m adversarial_gv --help
```

Collect 30 GSM8K samples that the fixed initial `gpt-4o-mini` Generator gets
wrong on its first and only answer (no V, backward pass, or prompt update):

```bash
python scripts/collect_gsm8k_initial_wrong.py --train-target 30 --val-target 15
```

The resumable collector writes `state.json`, `wrong_samples.jsonl`, and
`wrong_samples.csv` under `runs/gsm8k_initial_wrong/` and stops immediately at
the requested number of wrong samples. `split` preserves the original GSM8K
source split, while `collection_split` assigns the new hard set to train/val.

Each run writes its complete configuration and trajectory to a timestamped JSON
file under `runs/`. It also appends easy-to-scan initial, iteration, final, and
optional check rows to `runs/results.csv`.
TextGrad evaluator, backward-gradient, reduction, and optimizer calls are also
written one call per row to `runs/gradient_traces.csv` for auditing.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Vanilla TextGrad GSM8K baseline

Two separate TextGrad baselines are available.

`scripts/run_qwen4b_vanilla_textgrad_qwen32b.py` is a G-side ablation of the
completed GVGAN run. It removes the Verifier and alternating objective while
retaining the same immutable Generator system rules, trainable strategy,
Generator optimizer constraints, 512-token backward limit, sequential batch
size 6 schedule (35 updates over all 208 training cases), and initial/per-batch
train/val/test evaluations. The remaining objective is TextGrad's GSM8K
final-integer equality. Qwen3-4B is the forward model; Qwen3-32B supplies both
textual backward feedback and optimizer updates.

The run is resumable from `state.json`. Raw backward and optimizer prompts are
written to `textgrad_calls.jsonl`. Updates are unconditional, as in GVGAN; all
split evaluations are observational and never select or roll back a prompt.

```bash
python scripts/run_qwen4b_vanilla_textgrad_qwen32b.py
```

`scripts/run_qwen4b_original_textgrad_qwen32b.py` instead follows the original
`textgrad/evaluation/prompt_optimization.py` GSM8K protocol. It uses the
original monolithic trainable prompt, batch size 3, seed 42 NumPy shuffling,
three epochs with four updates each, no constraints or momentum, no custom
backward cap, non-decreasing validation selection, and initial/per-update
val/test evaluation. Only the data and engines are replaced by our balanced
CSV, Qwen3-4B forward, and Qwen3-32B backward/optimizer.

```bash
python scripts/run_qwen4b_original_textgrad_qwen32b.py
```
