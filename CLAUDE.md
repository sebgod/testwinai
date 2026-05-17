# testwinai — project context

Small .NET 10 / C# 14 / AOT console app for exercising Windows on-device AI on a
Snapdragon X Elite (X1E80100, Qualcomm Hexagon NPU). The codebase is intentionally
small — one Program.cs, one csproj. Treat it as a sandbox / probe, not a product.

## Hardware / OS preconditions (load-bearing)

These are required for anything in this repo to actually exercise the NPU:

- Snapdragon X Elite / X2 with Hexagon NPU (`Get-PnpDevice` shows
  `Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Hexagon(TM) NPU` in class
  `ComputeAccelerator`, status `OK`).
- Qualcomm NPU driver in the 30.0.219+ family (Copilot+ certified).
- Windows 11 24H2+ (this box is 26200 Canary).
- WinAppSDK 2.0.1 runtime installed machine-wide (the unpackaged app uses
  `Microsoft.WindowsAppRuntime.Bootstrap` to load it).
- System-staged Qualcomm QNN EP package:
  `MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_2.2420.44.0_arm64...`.
  This is provisioned automatically on Copilot+ PCs. The
  `ExecutionProviderCatalog.EnsureAndRegisterCertifiedAsync` call in `Program.cs`
  brings it from `NotReady` to `Ready` on first use.

Non-Snapdragon hardware: the QNN EP won't have an HTP backend. The code will
still build and the readiness/EP-catalog blocks will run, but `--model` will
either fail at load or fall back to CPU. Don't waste time chasing NPU behavior
on Intel/AMD/non-X-Elite ARM.

## Status quo — what works, what is permanently blocked here

**Working path (use this):**
- ORT-GenAI + QNN EP, direct NPU inference. Subcommands resolve the bundle
  via a small registry (see `Program.cs` → `ModelEntry` / `KnownModels`):
  - **default**: `llmware/qwen2.5-7b-instruct-onnx-qnn` (Qwen2.5 7B int4, ChatML)
  - **`--thinking`**: `llmware/deepseek-r1-distill-qwen-7b-onnx-qnn`
    (R1-distill, `<｜User｜>` / `<｜Assistant｜>` markers, EOS 151643)
- Phi-3.5 mini was the original target — ditched for chronic rambling.
- Trailing-duplicate-token stutter (`Yes..`, `bonjour le monde monde`, `3911`)
  is an int4-lm_head pathology, mitigated in code by `StutterGuard` (one-token
  lookahead, drops the duplicate before EOS). See `LIMITATIONS.md`.
- ~13 tok/s decode on both 7B bundles on the Hexagon HTP.

**Dead path on this box (don't chase):**
- The projected `Microsoft.Windows.AI.*` surface (LanguageModel/Phi Silica,
  TextRecognizer, ImageDescription, ImageScaler, ImageObjectExtractor,
  ImageObjectRemover) is permanently `CapabilityMissing`.
- Cause: corp policy in `HKLM:\Software\Policies\Microsoft\Windows\WindowsAI`
  sets `AllowRecallEnablement=0`, `AllowRecallExport=0`, `DisableClickToDo=1`.
  These gate the whole on-device AI feature provisioning umbrella on 26200
  Canary, not just Recall. The user cannot clear them (corp GPO).
- The original `DisableAIDataAnalysis=1` was removable and is gone; the three
  above are not.
- `Microsoft.Windows.AI.*` features are NOT classic FoDs on this build —
  `Get-WindowsCapability -Online` shows no entries for them even as
  `NotPresent`. They ship via Store-side AI Component Search gated on the
  Recall/Copilot+ umbrella. `Add-WindowsCapability` is a dead end.
- The readiness probe for the vision/imaging features remains in `Program.cs`
  as a diagnostic (prints `CapabilityMissing` and skips cleanly). The Phi Silica
  probe was removed — we use ORT-GenAI + QNN directly for text and there's no
  point round-tripping a known-blocked WinAI surface every run.

## Build & run

```powershell
# Build
dotnet build -c Release

# Diagnostics only (will report CapabilityMissing for the projected surface)
dotnet run -c Release

# One-time setup: fetch both bundles into C:\temp\testwinai-models
dotnet run -c Release -- download              # default (Qwen 2.5 7B)
dotnet run -c Release -- download --thinking   # DeepSeek-R1-Distill-Qwen-7B

# NPU inference — --model defaults to C:\temp\testwinai-models
dotnet run -c Release -- generate "<prompt>"               # Qwen
dotnet run -c Release -- generate --thinking "<prompt>"    # DeepSeek-R1
dotnet run -c Release -- chat                              # multi-turn REPL
dotnet run -c Release -- chat --thinking
dotnet run -c Release -- bench                             # default suite
dotnet run -c Release -- bench --thinking

# Advanced: point at a single bundle directly (template auto-detected)
dotnet run -c Release -- chat --model C:\path\to\some-bundle
```

`--model` can be either the models root dir (containing per-model subdirs by
slug — `qwen2.5-7b`, `deepseek-r1-7b`) or a single bundle dir containing
`genai_config.json` directly. Template is auto-detected from the bundle's
`tokenizer_config.json` (`bos_token` sniff); `--template chatml|deepseek-r1`
is the manual override.

The `bench` subcommand cross-products a list of system-prompt variants with a list
of tasks, runs each cell on the NPU, and writes:

- `bench-<timestamp>.json` — structured results (one record per cell with response,
  prefill/decode timings, stop reason, heuristic flags). Designed to be read by an
  LLM-as-judge in a later session for scoring/ranking.
- `bench-<timestamp>.md` — same data, human-readable side-by-side per task.

A custom suite is a JSON file with the shape:
```json
{
  "systemPrompts": [ { "id": "...", "text": "..." } ],
  "tasks":         [ { "id": "...", "prompt": "..." } ]
}
```

Forward slashes also work for the path (`C:/temp/models-qwen2.5-7b`) and are
needed when invoking from a bash-style shell (backslashes get eaten).

## Model bundles

Two known QNN ORT-GenAI bundles live under `C:\temp\testwinai-models\<slug>`:

| Slug | HF repo | Size | Family | Patches |
|---|---|---|---|---|
| `qwen2.5-7b` | `llmware/qwen2.5-7b-instruct-onnx-qnn` | 3.29 GB | Qwen2.5-Instruct, ChatML | none |
| `deepseek-r1-7b` | `llmware/deepseek-r1-distill-qwen-7b-onnx-qnn` | 3.34 GB | DeepSeek-R1-Distill-Qwen, `<｜User｜>` template | strip-backend-path |

Both share the same int8w/int16a transformer-on-NPU layout with int4
embeddings + lm_head on CPU. The bundles ship 4 × ~825 MB
`*_ctx_qnn.bin` / `*_cb_*.bin` EPContext shards — precompiled QNN HTP graphs
the CPU EP physically cannot execute, which itself is proof-of-NPU when the
model runs at non-CPU throughput.

The DeepSeek bundle's `genai_config.json` pins `backend_path: QnnHtp.dll` on
its prompt-processor and token-generator session_options, which blocks the
system-staged QNN EP. The `download` subcommand strips it (and keeps a
`.orig` backup). Adding a third bundle is a new `ModelEntry` in `KnownModels`
plus, if needed, a new case in `ApplyPatch`.

Any alternative model must be QNN-compiled for the Hexagon ISA on the target
machine. The bundled stubs are `QnnHtpV73Stub.dll` and `QnnHtpV81Stub.dll`
(v73 = older HTP, v81 = X1E/X2). A model compiled only for a different
Hexagon version will fail at load.

## NPU usage verification gotcha

Task Manager's NPU page (`Performance` → `NPU 0`) **only polls and graphs the
NPU counter while that tab is the foreground selection**. NPU activity that
happened while another tab was selected does NOT get retroactively backfilled.
So if a generation completes and the NPU graph still shows 0%, that's not a
sign QNN-direct isn't being measured — it's that the page wasn't actively
polling.

To verify NPU usage:
1. Select NPU 0 *before* starting the run.
2. Expect ~100% sustained Compute during decode, plus 100–300 MB in NPU
   Shared memory.
3. CPU should sit at single-digit % during decode (tokenizer + control flow
   only); GPU should be 0%.

## Log warnings that are benign

These appear on every successful run. Do not chase them:

- `QNN SetupBackend failed / Unable to load backend / load library failed`
  with source `onnxruntime-genai`. The OGA-bundled QNN EP probe fails (the
  shipped `onnxruntime_providers_qnn.dll` doesn't get its co-located
  `onnxruntime.dll` because of a build-asset-copy issue described below).
  Immediately after, the WinML-registered system QNN EP succeeds and runs
  the graph. Look for the `QNN EP only supports one device. Only the NPU
  device will be used.` line — that's the system EP winning.
- `Weight sharing only available with offline generation on x64 platform, not
  work on real device.` Weight sharing is a compile-time x64 optimization,
  not used at arm64 runtime; the .bin shards are already precompiled.
- `Config with key [ep.qnnexecutionprovider.*] already exists … will be
  overwritten`. The model's `genai_config.json` declares QNN options; our
  `Config.SetProviderOption("QNN", …)` calls then re-set identical values.
  Cosmetic.
- `Some nodes were not assigned to the preferred execution providers`.
  Expected: shape/control-flow ops live on CPU, MatMul/attention on NPU.

## Build-output quirk (`onnxruntime.dll` missing)

`Microsoft.ML.OnnxRuntime.QNN 1.24.4` ships `onnxruntime.dll` (31 MB) in its
`runtimes/win-arm64/native/` directory, but `dotnet build` is not copying it
into the project's output. All the Qualcomm `Qnn*.dll` files DO get copied.

The app runs anyway because WinAppSDK's Windows AI Runtime (provisioned by
`Microsoft.Windows.AI.MachineLearning 2.0.300`) provides a system-wide
`onnxruntime.dll` that the loader finds. This is why the WinML-registered QNN
EP path works end-to-end while the OGA-bundled one fails.

To fix (and silence the "load library failed" warning), force-copy the file
via an MSBuild item — but only if motivated; the WinML fallback is correct.

## Pinned package versions and why

- `Microsoft.ML.OnnxRuntimeGenAI.QNN` **0.13.1** — latest is 0.13.2 but it
  requires `Microsoft.ML.OnnxRuntime.QNN >= 1.25.1` which is not yet on
  NuGet (newest is 1.24.4). 0.13.1 pairs with 1.24.4 cleanly. Don't bump to
  0.13.2 until `ORT.QNN 1.25.x` ships.
- `Microsoft.WindowsAppSDK` 2.0.1 — kept for the projected AI surface
  diagnostics. Could be dropped entirely if we ever decide Phi Silica is
  truly out of scope.
- `Microsoft.Windows.AI.MachineLearning` 2.0.300 — provides
  `ExecutionProviderCatalog`, which is what stages the system QNN EP for
  use by our process.
- `Microsoft.ML.OnnxRuntime` was removed when we added `OnnxRuntimeGenAI.QNN`
  (its transitive `ORT.QNN 1.24.4` is the same runtime, plus the QNN EP).

## Workflow notes

- The session-level work log lives in `wip.md`. Update it after each
  meaningful change; it's allowed to rot. CLAUDE.md is the durable spec.
- Always verify NuGet package availability against the live registry
  (`api.nuget.org/v3-flatcontainer/<id>/index.json`) — don't rely on
  cached knowledge of which versions exist.
