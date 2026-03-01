# GLM-4.7 — Quality Eval Results

_Last updated: 2026-02-28_

---

## Summary

| Quant | HumanEval pass@1 | HLE | Notes |
|-------|:---:|:---:|-------|
| AWQ (QuantTrio/GLM-4.7-AWQ) | 12.2% | — | Significant quality loss from aggressive quantization |
| NVFP4 (Salyut1/GLM-4.7-NVFP4) | **43.9%** | 22% / **24%** | HumanEval: thinking off; HLE: without/with preserved thinking |
| NVFP4 + thinking on | 0% | — | Model generates prose instead of code in fewshot-multiturn format |
| FP8 (zai-org/GLM-4.7-FP8) | **45.0%** | 42.8% (official) | HumanEval: MTP off; MTP on = 32.3% (quality regression) |

HLE = Humanity's Last Exam (with tools). HumanEval = humaneval_instruct, pass@1, greedy.

---

## Eval infrastructure

Evals run via `scripts/eval_glm47.sh`:
- Routes through **buster-ripper** (`--eval-mode --eval-profile glm47`) on port 30002
- lm-eval `local-chat-completions`, `--apply_chat_template`, `--fewshot_as_multiturn`
- `gen_kwargs`: `max_tokens=4096, max_gen_toks=4096, repetition_penalty=1.05`
- 16 concurrent requests, greedy decoding (`do_sample=false`)

### buster-ripper `glm47` profile strategies

Three response transforms applied for GLM-4.7:

1. **`inject_chat_template_kwargs`** — injects `enable_thinking=false` into every
   request. Prevents the model from generating `<think>` blocks before code answers,
   which break lm-eval's `build_predictions_instruct` filter.

2. **`strip_code_fences`** — extracts code from ` ```python ... ``` ` blocks.
   GLM-4.7 wraps answers in markdown fences; lm-eval's filter drops the entire body
   if the response starts with ` ``` ` (`r[:r.find("```")]` == `""`). Searching
   anywhere in content (not just at start) handles the thinking-mode case where
   prose precedes the fence.

3. **`copy_reasoning_to_content`** — copies `reasoning_content` → `content` when
   content is empty (vLLM separates thinking tokens into a separate field when
   thinking is enabled; lm-eval only reads `content`).

---

## HumanEval investigation history

### Root cause of original 0% (thinking mode = default)

lm-eval's `humaneval_instruct` sends an assistant prefill ending with the function
signature. GLM-4.7 generates a **new response turn** wrapping its answer in a
markdown code block. The filter `build_predictions_instruct` does:

```python
doc["prompt"] + (r if r.find("```") == -1 else r[:r.find("```")])
```

Since the response **starts** with ` ```python `, `r[:0] == ""` — entire body
dropped. Fix: strip fences in buster-ripper before returning to lm-eval.

### Thinking on = 0% (persistent)

With `enable_thinking=true`, the model generates prose reasoning in the final
answer section rather than code. The thinking/code separation that works for
Claude Code clients does not produce scorable output in the lm-eval fewshot format.
`EVAL_THINKING` defaulted to `0` (off) after confirming no benefit.

### Repetition loops

Two HumanEval problems caused `!!!` generation loops hitting `max_tokens`.
Fixed with `repetition_penalty=1.05` (now default in `eval_glm47.sh`).

---

## FP8 model notes

`zai-org/GLM-4.7-FP8` — official Z.ai FP8 quantization.
- HumanEval: **45.0%** (MTP off) vs **32.3%** (MTP on) — MTP causes quality regression on FP8
- Official HLE (with tools): **42.8%** (different hardware/settings)
- Context limited to **88k** tokens on 4× RTX PRO 6000 96GB with `--kv-cache-dtype fp8 --gpu-memory-utilization 0.95`
- MTP disabled by default in `serve_glm47_fp8_vllm.sh` (`SPEC_TOKENS=0`)
- Same parsers: `--tool-call-parser glm47 --reasoning-parser glm45`
- Missing Triton MoE config for this GPU — see task #2 (tuning)
