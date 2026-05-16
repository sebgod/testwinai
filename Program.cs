using System.CommandLine;
using System.Diagnostics;
using System.Runtime.InteropServices;

using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.Windows.AI;
using Microsoft.Windows.AI.Imaging;
using Microsoft.Windows.AI.MachineLearning;
using Microsoft.Windows.AI.Text;
using Microsoft.Windows.ApplicationModel.DynamicDependency;

namespace TestWinAi;

// Subcommands:
//   probe                   machine info + Windows AI readiness probe + EP catalog dump
//                           (default when no subcommand given)
//   generate --model <dir>  one-shot Qwen2.5-family inference on the NPU via QNN HTP
//   chat     --model <dir>  interactive REPL with ChatML template + persistent KV cache
//
// The readiness probe runs at the top of every subcommand so it's obvious what's gated off
// vs. just not installed. On this corp-policy box every projected Windows AI feature reports
// CapabilityMissing because Recall is GPO-disabled (see CLAUDE.md).
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        // Silence ORT WARNING-level chatter (CLAUDE.md ⇒ all the "Config with key … will be
        // overwritten" / "Weight sharing only on x64" lines are benign). Set
        // ORT_LOG_SEVERITY_LEVEL=2 in the shell to re-enable.
        if (Environment.GetEnvironmentVariable("ORT_LOG_SEVERITY_LEVEL") is null)
            Environment.SetEnvironmentVariable("ORT_LOG_SEVERITY_LEVEL", "3");

        // WinAppSDK bootstrap — required by the readiness probe's WinRT projections.
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

        var root = new RootCommand("testwinai — Windows AI / Snapdragon X Elite NPU exploration.")
        {
            probeCmd, generateCmd, chatCmd,
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

        Console.WriteLine();
        Console.WriteLine("(tips:");
        Console.WriteLine("    testwinai generate --model <dir> \"<prompt>\"   one-shot NPU inference");
        Console.WriteLine("    testwinai chat     --model <dir>              multi-turn REPL with KV-cache)");
    }

    private static async Task DiagnosticsHeaderAsync()
    {
        PrintMachineInfo();
        DumpReadiness();
        await DumpEpCatalogAsync();
    }

    // Readiness of the projected Windows AI vision/imaging features — kept as a diagnostic
    // so it's obvious what's gated off. Phi Silica (LanguageModel) is intentionally not
    // probed here: corp Recall policy keeps it permanently CapabilityMissing on this box
    // and we use ORT-GenAI + QNN directly for text generation.
    private static void DumpReadiness()
    {
        Console.WriteLine();
        Console.WriteLine("== Windows AI feature readiness ==");
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

    // ---------- ORT-GenAI on the QNN execution provider (direct NPU path) ----------
    //
    // Loads an ORT-GenAI bundle (genai_config.json + ONNX + tokenizer + precompiled QNN HTP
    // context_*_ctx_qnn.bin shards) and runs generation through the QNN EP / HTP backend on
    // the Hexagon NPU. The model MUST be QNN-prepared — FP32/FP16 ONNX will silently fall
    // back to CPU on a Snapdragon X target. See CLAUDE.md for the model bundle expectations.

    // ChatML tokens, hard-coded since the format is stable across Qwen2 / Qwen2.5 / Qwen3.
    private const string ChatMlUserOpen = "<|im_start|>user\n";
    private const string ChatMlUserClose = "<|im_end|>\n";
    private const string ChatMlAssistantOpen = "<|im_start|>assistant\n";
    private const string ChatMlAssistantClose = "<|im_end|>\n";
    private const string ChatMlSystemOpen = "<|im_start|>system\n";
    private const string ChatMlSystemClose = "<|im_end|>\n";

    // Load model + tokenizer for any QNN ORT-GenAI bundle. We force QNN/HTP even though
    // genai_config.json already declares it — keeps behavior deterministic across bundles.
    private static (Model model, Tokenizer tokenizer, long loadMs) LoadQnnModel(string modelDir)
    {
        var sw = Stopwatch.StartNew();
        using var config = new Config(modelDir);
        config.ClearProviders();
        config.AppendProvider("QNN");
        config.SetProviderOption("QNN", "backend_type", "htp");
        // Off, or QNN's per-token ExtractBackendProfilingInfo floods stderr with
        // "ETW enabled previously, but disabled now" once WinML opens its own counters.
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

    // Interactive REPL. Holds one Generator alive across turns so the KV cache is reused —
    // each turn only encodes the new user message and AppendTokenSequences appends to the
    // existing cache rather than re-prefilling the full history.
    //
    // Slash commands: /exit /quit /clear /stats /help

    // Qwen2.5 <|im_end|>. Defensive hard-stop: when repetition_penalty suppresses it enough
    // that OGA's IsDone() doesn't trigger, we bail on this id ourselves. maxTokensPerTurn
    // is a second belt-and-braces cap so a runaway turn can't drain the session budget.
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
                gparams.SetSearchOption("do_sample", temperature > 0.0);
                gparams.SetSearchOption("temperature", temperature);
                gparams.SetSearchOption("top_p", topP);
                // Re-apply rather than rely on genai_config defaults so --repetition-penalty
                // and --no-repeat-ngram-size on the CLI actually take effect.
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
                    // when CancelKeyPress set e.Cancel=true. Re-prompt instead of exiting if
                    // the handler fired; treat true EOF (handler didn't fire) as exit.
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
                        // Ctrl+C between tokens — unreliable under PowerShell + AOT; Esc
                        // (below) is the dependable interrupt path.
                        if (interruptRequested) { interrupted = true; break; }

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

                        // Defensive EOS — see QwenImEndToken doc above.
                        if (lastToken == QwenImEndToken)
                            break;

                        var decoded = stream.Decode(lastToken);
                        Console.Write(decoded);
                        tokensThisTurn++;

                        // Whitespace-collapse detector: small int4 models on QNN HTP can fall
                        // into an emit-only-newlines loop because newlines feature in the
                        // chat template and look "expected" to the sampler. Bail out fast.
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
                    // Close the assistant turn so the KV cache stays well-formed — an
                    // unterminated <|im_start|>assistant confuses the next user turn.
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

    private static void PrintMachineInfo()
    {
        Console.WriteLine("== Machine ==");
        Console.WriteLine($"OS:       {RuntimeInformation.OSDescription}");
        Console.WriteLine($"Arch:     {RuntimeInformation.OSArchitecture} / process {RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($".NET:     {RuntimeInformation.FrameworkDescription}");
        Console.WriteLine($"Runtime:  {RuntimeInformation.RuntimeIdentifier}");
    }
}
