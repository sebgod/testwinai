# testwinai

[![CI](https://github.com/sebgod/testwinai/actions/workflows/dotnet-desktop.yml/badge.svg)](https://github.com/sebgod/testwinai/actions/workflows/dotnet-desktop.yml)

A small .NET 10 / C# 14 sandbox for exercising **Windows on-device AI on
Snapdragon X Elite** (Qualcomm Hexagon NPU). Runs 7B-class LLMs directly
on the NPU via the QNN execution provider + ORT-GenAI — no Phi Silica /
Copilot+ projected APIs required (those are gated by corp policy on the
dev box anyway; see [LIMITATIONS.md](LIMITATIONS.md)).

This is a probe, not a product. One `Program.cs`, one `.csproj`, AOT-compiled.

## What works

- **Direct NPU inference** at ~13 tok/s decode on the Hexagon HTP via
  Microsoft.ML.OnnxRuntimeGenAI.QNN 0.13.1 / ORT.QNN 1.24.4.
- Two pre-bundled QNN models, swappable with `--thinking`:
  - **`llmware/qwen2.5-7b-instruct-onnx-qnn`** — Qwen 2.5 7B instruct, ChatML template (default).
  - **`llmware/deepseek-r1-distill-qwen-7b-onnx-qnn`** — DeepSeek-R1-Distill-Qwen 7B, `<｜User｜>`/`<｜Assistant｜>` template (use `--thinking`).
- **Subcommands**: `download`, `generate`, `chat` (alt-screen TUI REPL with markdown rendering), `bench` (writes JSON + Markdown side-by-side comparisons).
- **Stutter mitigation** for an int4-`lm_head` pathology that emits
  trailing duplicate tokens (e.g. `Yes..`, `3911` instead of `391`) — fixed
  via one-token lookahead in the decode loop.
- **Sixel display-math rendering** in the chat TUI (STIX2Math.otf bundled).

## What doesn't work on this box (and why)

The projected `Microsoft.Windows.AI.*` surface (LanguageModel / Phi Silica,
TextRecognizer, ImageDescription, image scaling/extracting/removal) reports
`CapabilityMissing`. Cause: corp GPO under
`HKLM:\Software\Policies\Microsoft\Windows\WindowsAI` (`AllowRecallEnablement=0`,
`AllowRecallExport=0`, `DisableClickToDo=1`) gates the entire on-device AI
provisioning umbrella on Windows 11 26200 Canary, not just Recall. The
ORT-GenAI + QNN direct path is unaffected. See [LIMITATIONS.md](LIMITATIONS.md)
for the full triage.

## Requirements

- **Hardware**: Snapdragon X Elite / X2 with Hexagon NPU. The QNN HTP
  backend is fixed-function and ISA-specific — bundles compiled for the
  wrong Hexagon version fail at load. Stubs shipped: `QnnHtpV73Stub.dll`
  (older HTP), `QnnHtpV81Stub.dll` (X1E / X2).
- **Driver**: Qualcomm NPU driver 30.0.219+ (Copilot+ certified).
- **OS**: Windows 11 24H2+ (built/tested on 26200 Canary).
- **WinAppSDK 2.0.1** runtime installed machine-wide (the unpackaged app
  uses the bootstrap to load it).
- **System-staged Qualcomm QNN EP package** —
  `MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_…_arm64…`, auto-provisioned
  on Copilot+ PCs. First run brings it from `NotReady` to `Ready` via
  `ExecutionProviderCatalog.EnsureAndRegisterCertifiedAsync`.
- **.NET 10 SDK** to build from source.

Non-Snapdragon hardware is a dead end — the QNN EP has no HTP backend
elsewhere. `--model` either fails at load or falls back to CPU.

## Getting started

### Download a release

Pre-built binaries are on the [Releases](https://github.com/sebgod/testwinai/releases)
page. Only one RID ships because the QNN runtime is arm64-only:

| Platform | Asset |
|---|---|
| Windows ARM64 (Snapdragon X) | `testwinai-win-arm64.tar.gz` |

### Build from source

```powershell
git clone https://github.com/sebgod/testwinai.git
cd testwinai
dotnet build -c Release
```

Then fetch the model bundles once (~3.3 GB each, dropped under
`C:\temp\testwinai-models\<slug>`):

```powershell
dotnet run -c Release -- download              # default (Qwen 2.5 7B)
dotnet run -c Release -- download --thinking   # DeepSeek-R1-Distill-Qwen-7B
```

### Run inference

```powershell
# Diagnostics only (prints CapabilityMissing for the projected WinAI surface and exits)
dotnet run -c Release

# One-shot generation
dotnet run -c Release -- generate "Explain entropy in one sentence."
dotnet run -c Release -- generate --thinking "Is 131 prime? Show your reasoning."

# Multi-turn alt-screen TUI REPL with markdown rendering
dotnet run -c Release -- chat
dotnet run -c Release -- chat --thinking

# Bench suite (cross-products system prompts × tasks; emits bench-<ts>.{json,md})
dotnet run -c Release -- bench
dotnet run -c Release -- bench --thinking
```

`--model <path>` points at either the models root dir (with per-slug
subdirs) or a single bundle dir with `genai_config.json`. Template is
auto-detected from `tokenizer_config.json`; `--template chatml|deepseek-r1`
overrides.

## Verifying NPU usage

Task Manager's **NPU 0** page only polls while it's the foreground tab —
activity that happened while another tab was selected is **not** backfilled.
To confirm offload:

1. Select **NPU 0** before starting the run.
2. Expect ~100% sustained Compute during decode, plus 100–300 MB in NPU
   Shared memory.
3. CPU sits at single-digit % during decode (tokenizer + control flow only);
   GPU is 0%.

## Project structure

A single console project — no test project, no library split.

| File | Purpose |
|---|---|
| `Program.cs` | All subcommands (`download` / `generate` / `chat` / `bench`), `ModelEntry` registry, `StutterGuard`, EP catalog bootstrap |
| `testwinai.csproj` | Net 10 + WinAppSDK 2.0.1, AOT, partial trim, win-arm64 RID |
| `Directory.Packages.props` | Central package management with floating versions (Console.Lib tracks `2.12.*`) |
| `Directory.Build.props` | Auto-detects a sibling `sharpastro/Console.Lib` working tree and switches to a `ProjectReference` for in-tree iteration |
| `Fonts/STIX2Math.otf` | Bundled OpenType math font for sixel display-math (SIL OFL) |
| `CLAUDE.md` | Durable project spec — hardware preconditions, model bundles, build/run, log-warning triage |
| `LIMITATIONS.md` | Model pathologies (stutter, knowledge ranking, primality), the blocked WinAI surface, build-output quirks |
| `wip.md` | Session work log (allowed to rot) |

### Key dependencies

| Package | Purpose |
|---|---|
| [`Microsoft.ML.OnnxRuntimeGenAI.QNN`](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.QNN) | ORT-GenAI with the QNN EP for direct NPU inference (transitively pulls `Microsoft.ML.OnnxRuntime.QNN`) |
| [`Microsoft.WindowsAppSDK`](https://www.nuget.org/packages/Microsoft.WindowsAppSDK) | WinAppSDK bootstrap + projected `Microsoft.Windows.AI.*` surface (the latter is blocked on this box, kept for diagnostics) |
| [`Microsoft.Windows.AI.MachineLearning`](https://www.nuget.org/packages/Microsoft.Windows.AI.MachineLearning) | `ExecutionProviderCatalog` — stages the system QNN EP into the process |
| [`Console.Lib`](https://github.com/SharpAstro/Console.Lib) | Alt-screen TUI for `chat`: `MarkdownRenderer`, `VirtualTerminal`, `TerminalLayout`, `TextInputBar` |
| [`System.CommandLine`](https://www.nuget.org/packages/System.CommandLine) | Subcommand parsing |

## Further reading

- [CLAUDE.md](CLAUDE.md) — durable project spec
- [LIMITATIONS.md](LIMITATIONS.md) — model pathologies & blocked surfaces
