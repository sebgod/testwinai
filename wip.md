# testwinai — WIP

Small .NET 10 / C# 14 / AOT console app that exercises Windows AI on a Snapdragon X
Elite (X1E80100, Qualcomm Hexagon NPU).

## Pivot (2026-05-16, post-reboot)

The projected `Microsoft.Windows.AI.*` surface (Phi Silica + the five Imaging features)
is **dead on this box and cannot be revived without unblocking Recall**. The original
Group Policy that nuked it (`DisableAIDataAnalysis`) was removable, but corp has
`AllowRecallEnablement = 0`, `AllowRecallExport = 0`, `DisableClickToDo = 1` set in
`HKLM:\Software\Policies\Microsoft\Windows\WindowsAI` and those are not user-clearable.
On 26200 Canary the on-device AI feature payloads are not delivered as classic Features
on Demand at all (`Get-WindowsCapability -Online` lists no `Microsoft.Windows.AI.*`
entries even as `NotPresent`); they come through AI Component Search / Store-side
provisioning which is gated on the same Recall/Copilot+ umbrella. So
`Add-WindowsCapability` is a dead end too.

**Pivoted to direct NPU inference via ORT-GenAI + QNN EP.** The QNN execution provider
is fully functional on this box — `ExecutionProviderCatalog` resolves it to the staged
Qualcomm `onnxruntime_providers_qnn.dll`, and the NPU device shows OK in PnP. So we
just bypass the projected Windows AI APIs entirely and target the Hexagon HTP through
ORT-GenAI directly.

## What's built

- `testwinai.csproj` — net10.0-windows10.0.26100.0, `PublishAot=true`, win-arm64,
  unpackaged (`WindowsPackageType=None`).
- Packages:
  - `Microsoft.WindowsAppSDK` **2.0.1** — Phi Silica + Imaging AI APIs (kept for the
    readiness probe; those features still report `CapabilityMissing` and are skipped).
  - `Microsoft.Windows.AI.MachineLearning` **2.0.300** — `ExecutionProviderCatalog`,
    used to stage the QNN EP into ORT's plugin registry.
  - `Microsoft.ML.OnnxRuntimeGenAI.QNN` **0.13.1** — ORT-GenAI managed bindings +
    bundled QNN EP native (transitively pulls `Microsoft.ML.OnnxRuntime.QNN 1.24.4`
    and `Microsoft.ML.OnnxRuntimeGenAI.Managed 0.13.1`).
    - Note: 0.13.2 is on NuGet but wants `ORT.QNN 1.25.1` which is not yet shipped.
      Pinned to 0.13.1 + 1.24.4 for now.
- `Program.cs`:
  1. Machine info dump.
  2. Readiness probe over every projected AI feature (all `CapabilityMissing` here).
  3. `ExecutionProviderCatalog` pre- and post-stage dump
     (calls `EnsureAndRegisterCertifiedAsync` — QNN EP goes Ready/Certified).
  4. If `--model <dir>` provided → ORT-GenAI direct NPU path:
     - `Config.ClearProviders()` + `AppendProvider("QNN")` + `backend_type=htp`
     - Loads `Model`, `Tokenizer`, runs streamed generation, prints
       first-token latency + steady-state tok/s.
  5. Else → tries Phi Silica via `LanguageModel` (will report `CapabilityMissing`
     and skip, as documented above).
  6. Optional `--image <path>` runs `TextRecognizer` + `ImageDescriptionGenerator`
     (also `CapabilityMissing`).

## CLI

```text
testwinai                              # readiness + EP catalog + (futile) Phi Silica attempt
testwinai --model <dir>                # NPU inference on a Phi-3/Phi-3.5 ORT bundle via QNN EP
testwinai --model <dir> "<prompt>"     # ditto, with a custom prompt
testwinai --image <path>               # also tries OCR / image description (will skip)
testwinai "<prompt>"                   # custom prompt → Phi Silica (will skip)
```

## What works today

- Build clean (0 warnings, 0 errors) against WinAppSDK 2.0.1 + ORT-GenAI.QNN 0.13.1 +
  .NET 10.0.4 SDK.
- QNN EP path fully functional:
  - Detects `QNNExecutionProvider`, stages it (~44 s first run, ~50 ms cached).
  - Post-stage state: `Ready`, certified, pointing at
    `C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_2.2420.44.0_arm64__8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_qnn.dll`.
  - First staging produces a visible NPU spike in Task Manager.
- ORT-GenAI native binaries load successfully in-process. With no `--model` arg, the
  `Config`/`Model` constructors aren't called, so we don't know yet whether a real
  QNN-compiled model will run end-to-end.

## What doesn't work yet (and won't on this box)

- All six projected Windows AI features report `CapabilityMissing` and will keep doing
  so as long as Recall is corp-policy-blocked. Don't chase this further.

## Next step: get a model and run it

You need an ORT-GenAI model bundle compiled for QNN HTP. Layout expected by OGA:

```
<modelDir>\
  genai_config.json
  model.onnx                    (+ model.onnx.data if external-weights)
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

For Snapdragon X Elite (Hexagon HTP), the model **must** be QNN-prepared (INT8/INT16
QDQ quantized, ideally an EPContext-wrapped precompiled .bin). Pure FP32/FP16 models
will load but silently fall back to CPU.

**Recommended starting points** (each is several hundred MB to ~2 GB, one-time download):

1. **`microsoft/Phi-3.5-mini-instruct-onnx`** on Hugging Face
   - Several variants under different subfolders. Look for a `qnn/` subfolder if
     present; otherwise `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` will load
     on CPU EP only (useful as a sanity baseline, not NPU).
   - URL: <https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx>

2. **`microsoft/Phi-3-mini-4k-instruct-onnx`** on Hugging Face — older, more variants
   shipped including some QNN ones in past snapshots.
   - URL: <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx>

3. **Qualcomm AI Hub** — has Snapdragon-tuned bundles, requires free account.
   - URL: <https://aihub.qualcomm.com/>

Download steps (one option, using `huggingface-cli`):

```powershell
pip install -U huggingface_hub
huggingface-cli download microsoft/Phi-3.5-mini-instruct-onnx --include "qnn/*" --local-dir C:\models\Phi-3.5-mini-qnn
# (adjust --include glob to match whichever subfolder actually contains a QNN bundle)
```

Then run:

```powershell
cd C:\Users\SebastianGodelet\source\repos\sebgod\testwinai
dotnet run -c Release -- --model C:\models\Phi-3.5-mini-qnn "Explain the Hexagon NPU on the X1E80100."
```

Expected output:
- `load: NNNN ms` for the first-load (compiling/caching the QNN context, likely
  multi-second on first run, fast thereafter).
- Streamed token output to the console.
- Trailing line: `<N> tokens, first-token=<ms>, decode=<ms> (<tok/s>)`.
- NPU graph in Task Manager should spike during generation. If it stays flat and CPU
  spikes, the model wasn't actually QNN-compiled and the EP fell back to CPU.

## If the model loads but generation explodes / falls back to CPU

Most common causes:
1. Model isn't QNN-compiled. Check `genai_config.json` — the `execution_provider` /
   `provider_options` block usually names the intended EP. If it says `cpu` or `dml`,
   that bundle isn't for HTP.
2. `backend_type=htp` is being overridden by something in `genai_config.json`.
   The code calls `Config.ClearProviders()` before appending, which should win, but
   double-check.
3. ORT.QNN 1.24.4 vs the system-staged QNN EP (2.2420.44.0) version mismatch. Both
   should be ABI-compatible but if there are op-set conflicts, the EP will partition
   the graph and run unsupported nodes on CPU. ORT log level `verbose` will show
   partition decisions — enable via `Config.SetSearchOption` or env var
   `ORT_LOG_SEVERITY_LEVEL=0` before launch.

## Known gaps / future work

- AOT publish (`dotnet publish -c Release -r win-arm64`) not yet attempted. ORT-GenAI
  Managed is added as a `TrimmerRootAssembly` alongside ORT itself; AOT may still
  surface CsWinRT-side trim warnings from WinAppSDK projections that aren't
  exercised at runtime — easiest fix is to strip the unused WinAppSDK AI usings
  if Phi Silica is permanently dead here, but leaving for now since the readiness
  probe is still informative.
- No KV-cache warmup / batch / streaming chat — single-turn prompt only.
- `max_length=256` hardcoded; promote to a CLI flag once the basic path works.
- No content-filter customization needed (we're past the projected API, so the
  WinAppSDK content filter isn't in the loop here — model is what it is).

## Useful registry/PS one-liners

```powershell
# Verify which AI gate policies are still active
Get-ItemProperty HKLM:\Software\Policies\Microsoft\Windows\WindowsAI -EA SilentlyContinue

# List Windows AI components currently provisioned (will be empty / non-AI on this box)
Get-AppxPackage *AI* | Select Name, Version

# NPU device + driver status
Get-PnpDevice | Where-Object FriendlyName -like "*Hexagon*" | Format-Table FriendlyName, Class, Status

# AI services running (we expect WSAIFabricSvc + AppXSvc + StateRepository)
Get-Service | Where-Object Name -match "AI|Fabric" | Format-Table Name, Status, StartType
```

## Build / model state snapshot (as of 2026-05-16)

| Component                                | Status                                     |
| ---------------------------------------- | ------------------------------------------ |
| Build (Release, net10.0-windows arm64)   | 0 warn / 0 err                             |
| QNN EP registration                      | Ready, Certified                           |
| NPU device                               | OK                                         |
| WSAIFabricSvc                            | Running                                    |
| LanguageModel (Phi Silica)               | CapabilityMissing (corp Recall block)      |
| TextRecognizer / ImageDescription / ...  | CapabilityMissing (corp Recall block)      |
| ORT-GenAI native loads in-process        | Yes                                        |
| End-to-end NPU inference                 | **Working — 17.5 tok/s on Phi-3.5 mini**   |

## First NPU run (2026-05-16)

Model: `llmware/phi-3.5-mini-instruct-onnx-qnn`, placed at `C:\temp\models` (1.84 GB).
Note: bash mangles `C:\temp\models` (backslash-escape eating); use `C:/temp/models`
or run from cmd/pwsh. From cmd, native backslashes work fine.

```
dotnet run -c Release -- --model C:/temp/models "In one sentence, what is the Hexagon NPU?"
```

Result:
- `load: 23532 ms` (first run — QNN context binary deserialization + graph compile/cache)
- 192 tokens decoded, **17.5 tok/s** steady-state (4 × 460 MB context binaries running on
  the Hexagon HTP via the system-staged `onnxruntime_providers_qnn.dll`)
- Answer was coherent and on-topic ("Hexagon NPU is a specialized hardware accelerator
  designed for accelerating deep learning inference tasks…") before drifting into
  follow-up Q&A because the prompt wasn't wrapped in the Phi-3 chat template.

Warnings observed (all benign):
- One-time `QNN SetupBackend failed / Unable to load backend / load library failed`
  from `onnxruntime-genai` source — the OGA-bundled QNN probe fails first, then the
  WinML-registered system QNN EP takes over (`QNN EP only supports one device. Only
  the NPU device will be used.` immediately after).
- `Weight sharing only available with offline generation on x64 platform, not work
  on real device.` — weight-sharing is a compile-time x64 optimization, not used at
  arm64 runtime; the model is already pre-compiled (EPContext .bin shards).
- Multiple `Config with key [ep.qnnexecutionprovider.*] already exists … will be
  overwritten` — model's `genai_config.json` declares QNN options that we then
  re-set identically via our `Config.SetProviderOption` calls. Cosmetic.
- `Some nodes were not assigned to the preferred execution providers` —
  expected: shape/control-flow ops live on CPU, MatMul/attention land on NPU.

## Next polish (when motivated)

1. **Phi-3 chat template**: wrap user prompt in `<|user|>\n{p}<|end|>\n<|assistant|>\n`
   so generation stops cleanly at `<|end|>` and doesn't drift into hallucinated
   "Instruction 2" follow-ups. The bundle ships `chat_template.jinja`; the simplest
   fix is hard-coded prefix/suffix in `RunGenAiOnQnn` rather than pulling in Jinja.
2. **Cache the QNN context** so the 23 s first-load drops to <1 s on subsequent runs.
   QNN EP supports a `qnn_context_cache_enable` option pointing at a cache dir; wire
   it via `Config.SetProviderOption("QNN", "qnn_context_cache_enable", "1")`.
3. **Quiet the Config-already-exists warnings**: detect what `genai_config.json`
   already declares and only `SetProviderOption` for keys it doesn't. Or just live
   with them.
4. **Pin a `max_length`/`min_length`/`do_sample`/`temperature`/`top_p` CLI surface**
   instead of the hard-coded 256.
5. **Try AOT publish** (`dotnet publish -c Release -r win-arm64`). The OGA Managed
   assembly is a trimmer root, but CsWinRT trim warnings from unused WinAppSDK
   projections are still likely; easy fix would be to gate the projected-AI block
   behind a `--probe-winai` flag so the imports become optional.
