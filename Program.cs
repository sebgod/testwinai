using System.CommandLine;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;

using Microsoft.Graphics.Imaging;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.Windows.AI;
using Microsoft.Windows.AI.ContentSafety;
using Microsoft.Windows.AI.Imaging;
using Microsoft.Windows.AI.MachineLearning;
using Microsoft.Windows.AI.Text;
using Microsoft.Windows.ApplicationModel.DynamicDependency;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;

namespace TestWinAi;

// Subcommands:
//   probe                        machine info + Windows AI readiness probe + EP catalog dump
//                                (default when no subcommand given)
//   generate --model <dir>       one-shot Qwen2.5-family inference on the NPU via QNN HTP
//   chat     --model <dir>       interactive REPL with ChatML template + persistent KV cache
//   image    --image <path>      OCR + image description via the projected WinAppSDK Imaging
//                                features (gated by Recall policy — will report
//                                CapabilityMissing on a corp-locked box; see CLAUDE.md)
//
// The Windows AI feature readiness probe still runs at the top of every subcommand so it's
// obvious what's gated off vs. what's just not installed. On this dev box every projected
// feature lives at CapabilityMissing because corp policy disables Recall (see CLAUDE.md).
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        // Silence ORT WARNING-level logging. Genuine ORT errors (severity >= 3) still print.
        // Without this, every successful run dumps a screenful of "Config with key X already
        // exists … will be overwritten", "Weight sharing only available on x64", and the
        // benign QNN multi-device probe — all of which are documented as benign in CLAUDE.md.
        // To re-enable, set ORT_LOG_SEVERITY_LEVEL=2 (or 1/0) in the launching shell before
        // running, which overrides this default.
        if (Environment.GetEnvironmentVariable("ORT_LOG_SEVERITY_LEVEL") is null)
            Environment.SetEnvironmentVariable("ORT_LOG_SEVERITY_LEVEL", "3");

        // WinAppSDK bootstrap once for the whole process — the WinRT projections used by the
        // readiness probe + imaging path need this. Chat/generate don't strictly need it,
        // but the readiness probe runs before them, so initialize unconditionally.
        Bootstrap.Initialize(0x00020000, "");
        try
        {
            return await BuildRootCommand().Parse(args).InvokeAsync();
        }
        finally
        {
            Bootstrap.Shutdown();
        }
    }

    // ---------- CLI wiring ----------

    private static RootCommand BuildRootCommand()
    {
        var modelOption = new Option<DirectoryInfo>("--model", "-m")
        {
            Description = "Directory containing an ORT-GenAI Qwen2.5 model bundle " +
                          "(genai_config.json + ONNX + QNN context .bin shards + tokenizer).",
        };

        var maxLengthOption = new Option<int>("--max-length")
        {
            Description = "Generation cap (total sequence length including prompt + reply).",
            DefaultValueFactory = _ => 1024,
        };

        var temperatureOption = new Option<double>("--temperature")
        {
            Description = "Sampling temperature (chat). 0 = greedy.",
            DefaultValueFactory = _ => 0.7,
        };

        var topPOption = new Option<double>("--top-p")
        {
            Description = "Nucleus sampling p (chat).",
            DefaultValueFactory = _ => 0.9,
        };

        var systemOption = new Option<string?>("--system")
        {
            Description = "System message prepended to the chat. Default keeps Qwen2.5 brief; " +
                          "pass --system \"\" to disable, or --system \"…\" to override entirely.",
            DefaultValueFactory = _ =>
                "You are a concise assistant. Keep replies focused. " +
                "Avoid unsolicited follow-up questions and don't restate the prompt.",
        };

        var promptArgument = new Argument<string[]>("prompt")
        {
            Description = "Prompt words (joined with spaces). Optional — has a default.",
            Arity = ArgumentArity.ZeroOrMore,
        };

        var imageOption = new Option<FileInfo>("--image", "-i")
        {
            Description = "Path to a PNG/JPG to run OCR + image description on.",
            Required = true,
        };

        // probe
        var probeCmd = new Command("probe", "Machine info + Windows AI readiness + ORT EP catalog.");
        probeCmd.SetAction(async (_, _) => await ProbeAsync());

        // generate
        var generateCmd = new Command("generate", "One-shot NPU inference on the supplied model bundle.")
        {
            modelOption, maxLengthOption, promptArgument,
        };
        generateCmd.SetAction(async (pr, _) =>
        {
            var dir = pr.GetValue(modelOption);
            if (dir is null) { Console.Error.WriteLine("--model is required."); return 1; }
            var promptParts = pr.GetValue(promptArgument) ?? Array.Empty<string>();
            var prompt = promptParts.Length == 0
                ? "In one sentence, what is the Qualcomm Hexagon NPU and what does Windows use it for?"
                : string.Join(' ', promptParts);
            await DiagnosticsHeaderAsync();
            RunOneShot(dir.FullName, prompt, pr.GetValue(maxLengthOption));
            return 0;
        });

        // chat
        var chatMaxOption = new Option<int>("--max-length")
        {
            Description = "Per-session cap on total tokens (prompt history + replies).",
            DefaultValueFactory = _ => 4096,
        };
        var repetitionPenaltyOption = new Option<double>("--repetition-penalty")
        {
            Description = "Anti-repetition penalty (1.0 = off). 1.05 is the gentlest value " +
                          "that helps without suppressing the model's own <|im_end|> token. " +
                          "Anything above ~1.1 risks penalizing EOS, which makes turns run away.",
            DefaultValueFactory = _ => 1.05,
        };
        var noRepeatNgramOption = new Option<int>("--no-repeat-ngram-size")
        {
            Description = "Forbid any N-gram from repeating within the output. 0 = off (default). " +
                          "Avoid setting > 0 for chat: it forbids the chat-template N-grams " +
                          "(<|im_end|>\\n<|im_start|>user\\n etc.) from reappearing and breaks multi-turn stopping.",
            DefaultValueFactory = _ => 0,
        };
        var maxPerTurnOption = new Option<int>("--max-tokens-per-turn")
        {
            Description = "Hard cap on tokens generated in a single assistant turn. " +
                          "Prevents runaway generation if EOS gets suppressed.",
            DefaultValueFactory = _ => 1024,
        };
        var chatCmd = new Command("chat", "Interactive Qwen2.5 chat REPL on the NPU.")
        {
            modelOption, chatMaxOption, maxPerTurnOption, temperatureOption, topPOption, systemOption,
            repetitionPenaltyOption, noRepeatNgramOption,
        };
        chatCmd.SetAction(async (pr, _) =>
        {
            var dir = pr.GetValue(modelOption);
            if (dir is null) { Console.Error.WriteLine("--model is required."); return 1; }
            await DiagnosticsHeaderAsync();
            RunChat(
                dir.FullName,
                pr.GetValue(chatMaxOption),
                pr.GetValue(maxPerTurnOption),
                pr.GetValue(temperatureOption),
                pr.GetValue(topPOption),
                pr.GetValue(systemOption),
                pr.GetValue(repetitionPenaltyOption),
                pr.GetValue(noRepeatNgramOption));
            return 0;
        });

        // image
        var imageCmd = new Command("image", "Run WinAppSDK OCR + image description on a file.")
        {
            imageOption,
        };
        imageCmd.SetAction(async (pr, _) =>
        {
            await DiagnosticsHeaderAsync();
            await RunImagingAsync(pr.GetValue(imageOption)!.FullName);
            return 0;
        });

        var root = new RootCommand("testwinai — Windows AI / Snapdragon X Elite NPU exploration.")
        {
            probeCmd, generateCmd, chatCmd, imageCmd,
        };

        // No subcommand → probe.
        root.SetAction(async (_, _) =>
        {
            await ProbeAsync();
            return 0;
        });

        return root;
    }

    // ---------- Probe (no model required) ----------

    private static async Task ProbeAsync()
    {
        PrintMachineInfo();
        DumpReadiness();
        await DumpEpCatalogAsync();

        // Best-effort Phi Silica attempt for completeness — will skip cleanly on a box
        // where the projected surface is gated off.
        await TryPhiSilicaAsync(
            "In one sentence, what is the Qualcomm Hexagon NPU and what does Windows use it for?");

        Console.WriteLine();
        Console.WriteLine("(tips:");
        Console.WriteLine("    testwinai generate --model <dir> \"<prompt>\"   one-shot NPU inference");
        Console.WriteLine("    testwinai chat     --model <dir>              multi-turn REPL with KV-cache");
        Console.WriteLine("    testwinai image    --image <path>             OCR / description (gated))");
    }

    // Shared header for generate/chat/image — same diagnostics as probe minus the Phi Silica
    // attempt (we already know it doesn't work here, and chat especially shouldn't pause to
    // ask the SDK whether it might).
    private static async Task DiagnosticsHeaderAsync()
    {
        PrintMachineInfo();
        DumpReadiness();
        await DumpEpCatalogAsync();
    }

    // ---------- Readiness probe across every projected AI feature ----------

    private static void DumpReadiness()
    {
        Console.WriteLine();
        Console.WriteLine("== Windows AI feature readiness ==");
        PrintState("LanguageModel        (Phi Silica)", SafeReadyState(LanguageModel.GetReadyState));
        PrintState("TextRecognizer       (OCR)       ", SafeReadyState(TextRecognizer.GetReadyState));
        PrintState("ImageDescription                 ", SafeReadyState(ImageDescriptionGenerator.GetReadyState));
        PrintState("ImageScaler          (super-res) ", SafeReadyState(ImageScaler.GetReadyState));
        PrintState("ImageObjectExtractor (seg-mask)  ", SafeReadyState(ImageObjectExtractor.GetReadyState));
        PrintState("ImageObjectRemover   (inpainting)", SafeReadyState(ImageObjectRemover.GetReadyState));

        static AIFeatureReadyState? SafeReadyState(Func<AIFeatureReadyState> get)
        {
            try { return get(); }
            catch { return null; }
        }

        static void PrintState(string label, AIFeatureReadyState? state)
            => Console.WriteLine($"  {label} : {(state is null ? "<throw>" : state.ToString())}");
    }

    // ---------- Phi Silica (best-effort, expected to skip on this box) ----------

    private static async Task TryPhiSilicaAsync(string prompt)
    {
        Console.WriteLine();
        Console.WriteLine("== Phi Silica ==");

        if (!await EnsureFeatureReadyAsync("Phi Silica",
                LanguageModel.GetReadyState,
                LanguageModel.EnsureReadyAsync))
            return;

        using var lm = await LanguageModel.CreateAsync();

        Console.WriteLine();
        Console.WriteLine("-- GenerateResponseAsync (streaming) --");
        Console.WriteLine($"Prompt: {prompt}");
        Console.Write("Response: ");

        var sw = Stopwatch.StartNew();
        var gen = lm.GenerateResponseAsync(prompt);
        gen.Progress = (_, partial) => Console.Write(partial);
        var result = await gen;
        sw.Stop();
        Console.WriteLine();
        Console.WriteLine($"  status={result.Status}, elapsed={sw.ElapsedMilliseconds} ms");
    }

    // ---------- ORT-GenAI on the QNN execution provider (direct NPU path) ----------
    //
    // Bypasses the projected Microsoft.Windows.AI.* surface entirely. Loads an ORT-GenAI
    // model bundle (genai_config.json + ONNX + tokenizer) from <modelDir> and runs
    // generation through the QNN EP with the HTP backend, which lands on the Hexagon NPU.
    //
    // Expected layout for <modelDir> (Qwen2.5 QNN bundle from e.g. llmware):
    //   genai_config.json
    //   embeddings.onnx + lm_head.onnx
    //   context_*_ctx_qnn.bin   (precompiled QNN HTP shards — CPU cannot execute these)
    //   tokenizer.json, merges.txt, vocab.json, tokenizer_config.json, special_tokens_map.json
    //   chat_template.jinja     (informational — we hard-code the ChatML template below)
    //
    // For a Snapdragon X Elite (Hexagon HTP) target the model MUST be QNN-prepared. Pure
    // FP32/FP16 models will load but silently fall back to CPU.

    // Qwen2.5 ChatML tokens — see chat_template.jinja in the bundle. We hard-code rather
    // than pulling in a Jinja templating engine since ChatML is stable across the Qwen2 /
    // Qwen2.5 family and most Qwen3 variants.
    private const string ChatMlUserOpen = "<|im_start|>user\n";
    private const string ChatMlUserClose = "<|im_end|>\n";
    private const string ChatMlAssistantOpen = "<|im_start|>assistant\n";
    private const string ChatMlAssistantClose = "<|im_end|>\n";
    private const string ChatMlSystemOpen = "<|im_start|>system\n";
    private const string ChatMlSystemClose = "<|im_end|>\n";

    // Load the model + tokenizer for any QNN ORT-GenAI bundle. Returns load time in ms so the
    // caller can print it. Hard-forces the QNN EP with HTP backend even if genai_config.json
    // already declares it — the redundant SetProviderOption calls trigger benign
    // "Config with key X already exists … will be overwritten" warnings on every run.
    private static (Model model, Tokenizer tokenizer, long loadMs) LoadQnnModel(string modelDir)
    {
        var sw = Stopwatch.StartNew();
        using var config = new Config(modelDir);
        config.ClearProviders();
        config.AppendProvider("QNN");
        config.SetProviderOption("QNN", "backend_type", "htp");
        // Explicitly off — otherwise QNN tries to ExtractBackendProfilingInfo per token and
        // floods stderr with "ETW enabled previously, but disabled now" errors when the
        // ETW provider state changes mid-generation (which it does on Windows 11 26200 by
        // default when WinML opens its own counters).
        config.SetProviderOption("QNN", "profiling_level", "off");

        var model = new Model(config);
        var tokenizer = new Tokenizer(model);
        sw.Stop();
        return (model, tokenizer, sw.ElapsedMilliseconds);
    }

    private static void RunOneShot(string modelDir, string prompt, int maxLength)
    {
        Console.WriteLine();
        Console.WriteLine("== ORT GenAI on QNN EP (one-shot) ==");
        Console.WriteLine($"Model dir: {modelDir}");

        if (!Directory.Exists(modelDir))
        {
            Console.WriteLine($"  Model directory does not exist: {modelDir}");
            return;
        }

        var (model, tokenizer, loadMs) = LoadQnnModel(modelDir);
        try
        {
            Console.WriteLine($"  load: {loadMs} ms");
            Console.WriteLine($"Prompt: {prompt}");
            Console.Write("Response: ");

            // Wrap in ChatML so the response stops cleanly at <|im_end|>
            // instead of drifting into hallucinated "Instruction 2 / Follow-up" sections.
            var wrapped = ChatMlUserOpen + prompt + ChatMlUserClose + ChatMlAssistantOpen;
            using var inputTokens = tokenizer.Encode(wrapped);

            using var generatorParams = new GeneratorParams(model);
            generatorParams.SetSearchOption("max_length", maxLength);

            using var generator = new Generator(model, generatorParams);
            generator.AppendTokenSequences(inputTokens);

            using var stream = tokenizer.CreateStream();
            var swGen = Stopwatch.StartNew();
            long firstTokenMs = 0;
            int tokenCount = 0;
            while (!generator.IsDone())
            {
                generator.GenerateNextToken();
                if (tokenCount == 0)
                    firstTokenMs = swGen.ElapsedMilliseconds;

                var sequence = generator.GetSequence(0);
                int lastToken = sequence[sequence.Length - 1];
                Console.Write(stream.Decode(lastToken));
                tokenCount++;
            }
            swGen.Stop();

            Console.WriteLine();
            var totalMs = swGen.ElapsedMilliseconds;
            var decodeMs = totalMs - firstTokenMs;
            var tokPerSec = tokenCount > 1 && decodeMs > 0
                ? (tokenCount - 1) * 1000.0 / decodeMs
                : 0.0;
            Console.WriteLine(
                $"  {tokenCount} tokens, first-token={firstTokenMs} ms, decode={decodeMs} ms ({tokPerSec:0.0} tok/s)");
        }
        finally
        {
            tokenizer.Dispose();
            model.Dispose();
        }
    }

    // Interactive REPL. Holds one Generator alive across turns so its KV cache is reused —
    // each turn we only encode the new user message (wrapped in ChatML) and call
    // AppendTokenSequences, which adds to the existing cache rather than re-prefilling the
    // full history. If a future OGA build makes that pattern unreliable, fall back to
    // recreating the Generator per turn.
    //
    // Slash commands:
    //   /exit, /quit       leave the REPL
    //   /clear             reset KV cache / start new conversation (keeps model loaded)
    //   /stats             print last-turn token count + tok/s
    //   /help              show these
    // Qwen2.5 <|im_end|> token id. Defensive hard-stop: when repetition_penalty suppresses
    // 151645 enough that the model emits a near-miss, OGA's IsDone() can fail to trigger
    // even though early_stopping is on. We bail explicitly on this id, and we also cap each
    // turn at maxTokensPerTurn so a single runaway generation can't drain the session's
    // max_length budget.
    private const int QwenImEndToken = 151645;

    private static void RunChat(
        string modelDir,
        int maxLength,
        int maxTokensPerTurn,
        double temperature,
        double topP,
        string? system,
        double repetitionPenalty,
        int noRepeatNgramSize)
    {
        Console.WriteLine();
        Console.WriteLine("== Chat (Qwen2.5 on QNN HTP) ==");
        Console.WriteLine($"Model dir: {modelDir}");

        if (!Directory.Exists(modelDir))
        {
            Console.WriteLine($"  Model directory does not exist: {modelDir}");
            return;
        }

        var (model, tokenizer, loadMs) = LoadQnnModel(modelDir);
        try
        {
            Console.WriteLine($"  load: {loadMs} ms");
            Console.WriteLine($"  max-length={maxLength}, max-tokens-per-turn={maxTokensPerTurn}, " +
                              $"temperature={temperature}, top-p={topP}, " +
                              $"rep-penalty={repetitionPenalty}, no-repeat-ngram={noRepeatNgramSize}");
            Console.WriteLine("  /exit  /clear  /stats  /help");
            Console.WriteLine("  Ctrl+C or Esc during generation = interrupt this turn. (PowerShell");
            Console.WriteLine("   sometimes forwards Ctrl+C as termination instead — use Esc if so.)");

            // Mid-generation interrupt: Ctrl+C cancels the current turn instead of killing
            // the process. Esc (polled between tokens) does the same. Both reset to false at
            // the start of each turn.
            bool interruptRequested = false;
            ConsoleCancelEventHandler cancelHandler = (_, e) =>
            {
                e.Cancel = true;
                interruptRequested = true;
            };
            Console.CancelKeyPress += cancelHandler;

            GeneratorParams gparams = null!;
            Generator generator = null!;
            int turnCount = 0;
            long lastTurnMs = 0;
            int lastTurnTokens = 0;

            void ResetSession()
            {
                generator?.Dispose();
                gparams?.Dispose();
                gparams = new GeneratorParams(model);
                gparams.SetSearchOption("max_length", maxLength);
                // Light sampling for chat: more natural responses without going off the rails.
                gparams.SetSearchOption("do_sample", temperature > 0.0);
                gparams.SetSearchOption("temperature", temperature);
                gparams.SetSearchOption("top_p", topP);
                // Anti-degenerate-loop guardrails. Even Qwen2.5 7B int4 can stumble into
                // catastrophic repetition (especially when it drifts into another language
                // mid-answer). The bundle's genai_config.json already sets
                // repetition_penalty=1.05 by default; we re-apply via search options so
                // --repetition-penalty on the CLI works as expected.
                gparams.SetSearchOption("repetition_penalty", repetitionPenalty);
                if (noRepeatNgramSize > 0)
                    gparams.SetSearchOption("no_repeat_ngram_size", noRepeatNgramSize);
                generator = new Generator(model, gparams);
                turnCount = 0;

                // System message, if any, gets prepended to the very first turn's prefill.
                if (!string.IsNullOrWhiteSpace(system))
                {
                    var systemTokens = tokenizer.Encode(ChatMlSystemOpen + system + ChatMlSystemClose);
                    try { generator.AppendTokenSequences(systemTokens); }
                    finally { systemTokens.Dispose(); }
                }
            }

            ResetSession();
            using var stream = tokenizer.CreateStream();

            while (true)
            {
                Console.WriteLine();
                Console.Write("you> ");
                var line = Console.ReadLine();
                if (line is null)
                {
                    // Ctrl+C at the prompt: .NET's ReadLine returns null on Windows even
                    // when our CancelKeyPress handler set e.Cancel=true. Distinguish that
                    // from a real EOF (stdin closed / piped-input exhausted) by checking
                    // whether the handler fired. If it did, swallow it and re-prompt.
                    if (interruptRequested)
                    {
                        interruptRequested = false;
                        Console.WriteLine();
                        continue;
                    }
                    break;
                }
                line = line.Trim();
                if (line.Length == 0) continue;

                if (line.Equals("/exit", StringComparison.OrdinalIgnoreCase) ||
                    line.Equals("/quit", StringComparison.OrdinalIgnoreCase))
                    break;
                if (line.Equals("/help", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("  /exit | /quit   leave");
                    Console.WriteLine("  /clear          reset conversation (model stays loaded)");
                    Console.WriteLine("  /stats          last-turn timing");
                    continue;
                }
                if (line.Equals("/clear", StringComparison.OrdinalIgnoreCase))
                {
                    ResetSession();
                    Console.WriteLine("  (conversation reset)");
                    continue;
                }
                if (line.Equals("/stats", StringComparison.OrdinalIgnoreCase))
                {
                    if (lastTurnTokens == 0) { Console.WriteLine("  (no turns yet)"); }
                    else
                    {
                        var tps = lastTurnTokens * 1000.0 / Math.Max(1, lastTurnMs);
                        Console.WriteLine($"  last turn: {lastTurnTokens} tokens in {lastTurnMs} ms ({tps:0.0} tok/s)");
                    }
                    continue;
                }

                // Encode just this turn's user wrapper. The KV cache from prior turns is
                // already inside `generator`; we only add the new tokens.
                var turnPrompt = ChatMlUserOpen + line + ChatMlUserClose + ChatMlAssistantOpen;
                using var turnTokens = tokenizer.Encode(turnPrompt);

                try
                {
                    generator.AppendTokenSequences(turnTokens);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  append failed ({ex.GetType().Name}: {ex.Message}); resetting session");
                    ResetSession();
                    continue;
                }

                Console.Write("ai>  ");
                interruptRequested = false;
                var swTurn = Stopwatch.StartNew();
                long firstTokenMs = 0;
                int tokensThisTurn = 0;
                int consecutiveBlankTokens = 0;
                bool capped = false;
                bool interrupted = false;
                bool collapsed = false;
                try
                {
                    while (!generator.IsDone())
                    {
                        // Ctrl+C arrived between tokens. Unreliable under PowerShell + AOT
                        // (PowerShell may bypass the .NET CancelKeyPress handler), but works
                        // in cmd / Windows Terminal in some configurations.
                        if (interruptRequested) { interrupted = true; break; }

                        // Esc key pressed during generation — the reliable interrupt path.
                        // ReadKey(intercept) consumes the key without echoing.
                        if (Console.KeyAvailable)
                        {
                            var k = Console.ReadKey(intercept: true);
                            if (k.Key == ConsoleKey.Escape) { interrupted = true; break; }
                        }

                        generator.GenerateNextToken();
                        if (tokensThisTurn == 0)
                            firstTokenMs = swTurn.ElapsedMilliseconds;

                        var sequence = generator.GetSequence(0);
                        int lastToken = sequence[sequence.Length - 1];

                        // Defensive EOS: OGA's IsDone respects early_stopping + eos_token_id,
                        // but repetition_penalty can suppress 151645 enough that the model
                        // emits something close instead and runs forever. Hard-stop on the
                        // real Qwen <|im_end|> id if we see it.
                        if (lastToken == QwenImEndToken)
                            break;

                        var decoded = stream.Decode(lastToken);
                        Console.Write(decoded);
                        tokensThisTurn++;

                        // Degenerate-output detector: small int4 models on QNN HTP sometimes
                        // collapse into an emit-only-whitespace loop (e.g. 450 consecutive
                        // newlines for a "write code" prompt). Newlines aren't penalized
                        // enough by repetition_penalty=1.05 because they appear all through
                        // the chat template, so they look "expected" to the sampler. If we
                        // see >16 consecutive whitespace-only tokens, the turn has collapsed.
                        if (string.IsNullOrWhiteSpace(decoded))
                        {
                            consecutiveBlankTokens++;
                            if (consecutiveBlankTokens > 16) { collapsed = true; break; }
                        }
                        else
                        {
                            consecutiveBlankTokens = 0;
                        }

                        if (tokensThisTurn >= maxTokensPerTurn)
                        {
                            capped = true;
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine();
                    Console.WriteLine($"  generation aborted: {ex.GetType().Name}: {ex.Message}");
                    ResetSession();
                    continue;
                }
                swTurn.Stop();
                Console.WriteLine();
                if (interrupted || collapsed)
                {
                    Console.WriteLine(collapsed
                        ? "  (output collapsed into whitespace — aborted; try rephrasing or /clear)"
                        : "  (interrupted)");
                    // Close the assistant turn properly so the KV cache stays well-formed
                    // for the next user turn — otherwise we'd have an unterminated
                    // <|im_start|>assistant section in history, which confuses Qwen on the
                    // next turn.
                    try
                    {
                        using var closeTokens = tokenizer.Encode(ChatMlAssistantClose);
                        generator.AppendTokenSequences(closeTokens);
                    }
                    catch { /* if append fails, /clear is the user's recovery */ }
                }
                else if (capped)
                {
                    Console.WriteLine($"  (turn hit --max-tokens-per-turn={maxTokensPerTurn} cap)");
                }

                lastTurnMs = swTurn.ElapsedMilliseconds;
                lastTurnTokens = tokensThisTurn;
                turnCount++;
                var decodeMs = lastTurnMs - firstTokenMs;
                var decodeTps = tokensThisTurn > 1 && decodeMs > 0
                    ? (tokensThisTurn - 1) * 1000.0 / decodeMs
                    : 0.0;
                Console.WriteLine(
                    $"  [turn {turnCount}: {tokensThisTurn} tok, prefill={firstTokenMs} ms, decode={decodeMs} ms ({decodeTps:0.0} tok/s)]");
            }

            Console.CancelKeyPress -= cancelHandler;
            generator?.Dispose();
            gparams?.Dispose();
        }
        finally
        {
            tokenizer.Dispose();
            model.Dispose();
        }
    }

    // ---------- Imaging: OCR + image description ----------

    private static async Task RunImagingAsync(string imagePath)
    {
        Console.WriteLine();
        Console.WriteLine($"== Imaging features (input: {imagePath}) ==");
        if (!File.Exists(imagePath))
        {
            Console.WriteLine($"  File not found: {imagePath}");
            return;
        }

        // Decode the file → SoftwareBitmap → ImageBuffer (the format every AI Imaging API takes).
        using var stream = new InMemoryRandomAccessStream();
        await using (var fs = File.OpenRead(imagePath))
        {
            var bytes = new byte[fs.Length];
            await fs.ReadExactlyAsync(bytes);
            await stream.WriteAsync(bytes.AsBuffer());
        }
        stream.Seek(0);
        var decoder = await BitmapDecoder.CreateAsync(stream);
        var bitmap = await decoder.GetSoftwareBitmapAsync(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
        var imageBuffer = ImageBuffer.CreateForSoftwareBitmap(bitmap);
        Console.WriteLine($"  Decoded {bitmap.PixelWidth}x{bitmap.PixelHeight} {bitmap.BitmapPixelFormat}");

        // OCR
        if (await EnsureFeatureReadyAsync("TextRecognizer",
                TextRecognizer.GetReadyState,
                TextRecognizer.EnsureReadyAsync))
        {
            Console.WriteLine();
            Console.WriteLine("-- TextRecognizer.RecognizeTextFromImageAsync --");
            var ocr = await TextRecognizer.CreateAsync();
            var sw = Stopwatch.StartNew();
            var recognized = await ocr.RecognizeTextFromImageAsync(imageBuffer);
            sw.Stop();
            Console.WriteLine($"  elapsed={sw.ElapsedMilliseconds} ms, text angle={recognized.TextAngle:0.0}°, lines={recognized.Lines.Length}");
            foreach (var line in recognized.Lines)
            {
                var avgConf = line.Words.Length == 0 ? 0f : line.Words.Average(w => w.MatchConfidence);
                Console.WriteLine($"    [{avgConf:P0}] {line.Text}");
            }
        }

        // Image-to-caption
        if (await EnsureFeatureReadyAsync("ImageDescriptionGenerator",
                ImageDescriptionGenerator.GetReadyState,
                ImageDescriptionGenerator.EnsureReadyAsync))
        {
            Console.WriteLine();
            Console.WriteLine("-- ImageDescriptionGenerator.DescribeAsync --");
            var describer = await ImageDescriptionGenerator.CreateAsync();
            var sw = Stopwatch.StartNew();
            var op = describer.DescribeAsync(imageBuffer, ImageDescriptionKind.DetailedDescription, new ContentFilterOptions());
            var desc = await op;
            sw.Stop();
            Console.WriteLine($"  elapsed={sw.ElapsedMilliseconds} ms, status={desc.Status}");
            Console.WriteLine($"  {desc.Description}");
        }
    }

    // ---------- ML Execution Provider catalog ----------

    private static async Task DumpEpCatalogAsync()
    {
        Console.WriteLine();
        Console.WriteLine("== ONNX Runtime Execution Provider catalog ==");
        try
        {
            var catalog = ExecutionProviderCatalog.GetDefault();

            var before = catalog.FindAllProviders().ToArray();
            Console.WriteLine($"Known providers (pre-stage): {before.Length}");
            foreach (var ep in before)
                PrintEp(ep);

            Console.WriteLine("Calling EnsureAndRegisterCertifiedAsync()…");
            var swEp = Stopwatch.StartNew();
            await catalog.EnsureAndRegisterCertifiedAsync();
            swEp.Stop();
            Console.WriteLine($"  done in {swEp.ElapsedMilliseconds} ms");

            var after = catalog.FindAllProviders().ToArray();
            Console.WriteLine($"Known providers (post-stage): {after.Length}");
            foreach (var ep in after)
                PrintEp(ep);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  EP catalog probe failed: {ex.GetType().Name}: {ex.Message}");
        }

        static void PrintEp(ExecutionProvider ep)
        {
            Console.WriteLine($"  - {ep.Name,-20} ready={ep.ReadyState} cert={ep.Certification}");
            if (!string.IsNullOrEmpty(ep.LibraryPath))
                Console.WriteLine($"      path: {ep.LibraryPath}");
            if (ep.PackageId is { FullName: { Length: > 0 } pkg })
                Console.WriteLine($"      pkg:  {pkg}");
        }
    }

    // ---------- Helpers ----------

    private delegate Windows.Foundation.IAsyncOperationWithProgress<AIFeatureReadyResult, double> EnsureReady();

    private static async Task<bool> EnsureFeatureReadyAsync(string name, Func<AIFeatureReadyState> getState, EnsureReady ensure)
    {
        var state = getState();
        Console.WriteLine($"  {name}.GetReadyState() = {state}");
        if (state == AIFeatureReadyState.Ready) return true;

        if (state != AIFeatureReadyState.NotReady)
        {
            Console.WriteLine($"  {name} is not usable here (state={state}). Skipping.");
            return false;
        }

        Console.WriteLine($"  Staging {name} (this may download model weights)…");
        var op = ensure();
        op.Progress = (_, p) => Console.Write($"\r    staging: {p,6:0.0}%   ");
        var ready = await op;
        Console.WriteLine();
        Console.WriteLine($"    Status={ready.Status}, ExtendedError=0x{ready.ExtendedError?.HResult:X8}");
        return ready.Status == AIFeatureReadyResultState.Success;
    }

    private static void PrintMachineInfo()
    {
        Console.WriteLine("== Machine ==");
        Console.WriteLine($"OS:       {RuntimeInformation.OSDescription}");
        Console.WriteLine($"Arch:     {RuntimeInformation.OSArchitecture} / process {RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($".NET:     {RuntimeInformation.FrameworkDescription}");
        Console.WriteLine($"Runtime:  {RuntimeInformation.RuntimeIdentifier}");
    }
}
