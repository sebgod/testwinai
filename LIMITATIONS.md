# Limitations & known pathologies

Status snapshot for `llmware/qwen2.5-7b-instruct-onnx-qnn` and
`llmware/deepseek-r1-distill-qwen-7b-onnx-qnn` on the Hexagon HTP via QNN EP
+ ORT-GenAI 0.13.1 / ORT.QNN 1.24.4 on this box. The bundle architecture is
identical between the two — int4 weights / fp32 activations on CPU for
embeddings + lm_head; int8w / int16a on the NPU for transformer blocks.

## Decode-time pathologies — mitigated

The trailing-duplicate-token stutter (`Yes..`, `bonjour le monde monde`,
`~597 million million`, `3911`, code `` `).`).` `` ) was an int4-quantized
lm_head artifact. The model's logits at the end of a short answer ranked a
"duplicate of the previous token" just above EOS, by a tiny margin. The
pathology is deterministic — survives `temperature=0.2`, survives
`repetition_penalty=1.0`, survives every system-prompt variant, and survives
a model swap (the DeepSeek bundle uses the same int4 lm_head precision and
showed the same stutter on its outputs, e.g. `\boxed{391}\]]`).

**Mitigated in code** (commit `904cfa2`, `StutterGuard` in `Program.cs`):
one-token lookahead in the decode loop. If the model would emit EOS right
after a token whose id equals the previously emitted id, the trailing
duplicate is dropped. Verified across the full bench suite on Qwen 2.5 7B:
every previously-stuttering cell now stops cleanly, and `17 × 23` recovers
the correct `391` instead of `3911` (the spurious trailing `1` was the bug
itself, not a cosmetic artifact).

### Why we believe it's the LM head (kept for the record)

- int4 over a 152K-token vocab gives ~5 bits of precision per logit
- Both bundles ship a separate `lm_head.onnx` / `*_head_quant.onnx` at ~340 MB,
  which matches `vocab × hidden × 0.5 bytes`, i.e. int4
- The pathology is deterministic across both samplers
- All five prompt variants in the v1 bench showed the same stutter on the
  same cells, ruling out the prompt layer
- The math overshoot (`391` vs `3911`) fits the same one-extra-token shape
- The model swap to DeepSeek-R1-Distill (same head precision) showed the
  same family of artifacts

### Hypotheses ruled out before we found the right fix

- **Repetition penalty suppressing EOS.** Tested at `rep=1.0`: stutter identical.
- **Sampling variance.** Tested at `temp=0.2`: stutter identical.
- **Two-append bug dropping the system prompt** (real bug, fixed in `7377149`):
  the stutter still appeared post-fix, so it wasn't the cause.
- **Prompt under-specification.** Five system-prompt variants
  (`current` / `strict` / `v2a-minimum-tokens` / `v2b-anti-stutter` /
  `v2c-format-aware`) all stuttered identically on the same cells.

## Model-level issues still present (NOT mitigable from our side)

| Issue | Example | Notes |
|---|---|---|
| Qwen 2.5 7B knowledge ranking | "Top 3 populous continents" → Asia / Africa / **N. America** | Europe (~745M) > N. America (~600M), but the model consistently picks NA. Training-data artifact. **DeepSeek-R1 fixes this** — it reasons through the ranking and gets Europe right. |
| Qwen 2.5 7B self-contradiction on primality | "Is 131 prime?" → "divisible by 1, 131, and no other numbers, but it is not a prime number" | Same response volunteers the correct divisor set then concludes the wrong way. Pattern-matched answer; the chat-style decoder doesn't notice the logical contradiction inside its own reply. **DeepSeek-R1 fixes this** — trial-divides by primes ≤ √131 and correctly concludes prime. New `prime` task in the default bench suite. |
| DeepSeek-R1 verbosity | `<think>` blocks of 100–400 tokens before every answer | Inherent to the R1-distill recipe. Trades brevity for reasoning quality. |
| DeepSeek-R1 number recall | "Asia ~1.4 billion" (real: ~4.7B) | Reasoning-based but doesn't override miscalibrated training-time knowledge. |
| Hexagon HTP is not general-purpose | No CUDA-style kernel authoring | Fixed-function tensor accelerator. Custom kernels would need Qualcomm's QNN SDK with HTP op authoring. |
| No QNN-compiled VL models on HF | Phi-3.5-vision, LLaVA, Qwen2.5-VL only as DML / OpenVINO bundles | Text-only on the NPU path here. |
| HTP-version specificity | `QnnHtpV73Stub.dll` (older) vs `QnnHtpV81Stub.dll` (X1E/X2) | A bundle compiled only for a different Hexagon ISA fails at load. |

## DeepSeek bundle EPContext gotcha — patched on download

The DeepSeek bundle's `genai_config.json` pins
`"backend_path": "QnnHtp.dll"` on the prompt-processor + token-generator
session_options. This routes the session to the OGA-bundled QNN EP, whose
co-located `onnxruntime.dll` isn't copied to the build output (see the
"Build-output quirk" section in `CLAUDE.md`). The OGA-bundled probe fails,
and the system-staged QNN EP refuses to claim a session already pinned to
the bundled path. Load throws:

```
Could not find an implementation for EPContext(1) node …
```

**Fix**: `testwinai download --thinking` strips `backend_path` from both
stages and keeps a `.orig` backup. The system QNN EP then takes over
normally. The Qwen bundle doesn't have this issue (no `backend_path` pinned).

## Projected `Microsoft.Windows.AI.*` surface — permanently blocked

On this box (corp-managed Snapdragon X Elite, Windows 11 26200 Canary):

- `LanguageModel` (Phi Silica), `TextRecognizer`, `ImageDescription`,
  `ImageScaler`, `ImageObjectExtractor`, `ImageObjectRemover` all report
  `CapabilityMissing`.
- Cause: GPO under `HKLM:\Software\Policies\Microsoft\Windows\WindowsAI`
  sets `AllowRecallEnablement=0`, `AllowRecallExport=0`, `DisableClickToDo=1`.
  These three gate the entire on-device AI provisioning umbrella, not just
  Recall. User cannot clear them.
- These features ship via Store-side AI Component Search and are NOT classic
  FoDs — `Get-WindowsCapability -Online` shows no entries.
- Net effect: only the ORT-GenAI + QNN direct path works on this machine.

## Build / runtime quirks (benign but noisy)

- `onnxruntime.dll` from `Microsoft.ML.OnnxRuntime.QNN 1.24.4` doesn't get
  copied to the build output by `dotnet build`. Process runs anyway because
  WinAppSDK's Windows AI Runtime provisions a system-wide `onnxruntime.dll`.
  Log warnings (`QNN SetupBackend failed`, `load library failed`) come from
  the OGA-bundled EP probe failing; the system-registered QNN EP wins
  immediately after. Confirm with `QNN EP only supports one device. Only the
  NPU device will be used.`
- `Weight sharing only available with offline generation on x64 platform`
  — x64-only compile-time optimization, not used at arm64 runtime.
- `Config with key [ep.qnnexecutionprovider.*] already exists … will be
  overwritten` — our `Config.SetProviderOption` calls re-apply identical
  values from `genai_config.json`. Cosmetic.
- `Some nodes were not assigned to the preferred execution providers` —
  expected; shape / control-flow ops live on CPU, MatMul / attention on NPU.

## Throughput

~13 tok/s decode on both bundles (Qwen 2.5 7B and DeepSeek-R1-Distill-Qwen-7B
share the same architecture). Prefill is unobservable for short prompts.
Task Manager's NPU 0 page must be selected **before** the run to graph
activity — it does not backfill.
