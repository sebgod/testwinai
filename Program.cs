using System.CommandLine;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

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
//   bench    --model <dir>  run a system-prompt × task suite, write JSON + markdown for
//                           offline scoring (use built-in suite or --suite <file>)
//
// The readiness probe runs at the top of every subcommand so it's obvious what's gated off
// vs. just not installed. On this corp-policy box every projected Windows AI feature reports
// CapabilityMissing because Recall is GPO-disabled (see CLAUDE.md).
internal static partial class Program
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

        // bench
        var benchSuiteOption = new Option<FileInfo?>("--suite")
        {
            Description = "Path to a JSON suite (systemPrompts + tasks). " +
                          "Omit to use the built-in default suite.",
        };
        var benchOutputOption = new Option<DirectoryInfo?>("--output", "-o")
        {
            Description = "Directory for bench-<timestamp>.{json,md}. Defaults to the current directory.",
        };
        var benchCmd = new Command(
            "bench",
            "Run a system-prompt × task suite and write JSON + markdown for offline scoring.")
        {
            modelOption, benchSuiteOption, benchOutputOption,
            temperatureOption, topPOption, repetitionPenaltyOption, maxPerTurnOption,
        };
        benchCmd.SetAction(async (pr, _) =>
        {
            var dir = pr.GetValue(modelOption);
            if (dir is null) { Console.Error.WriteLine("--model is required."); return 1; }
            await DiagnosticsHeaderAsync();
            RunBench(
                dir.FullName,
                pr.GetValue(benchSuiteOption)?.FullName,
                pr.GetValue(benchOutputOption)?.FullName ?? Environment.CurrentDirectory,
                pr.GetValue(temperatureOption),
                pr.GetValue(topPOption),
                pr.GetValue(repetitionPenaltyOption),
                pr.GetValue(maxPerTurnOption));
            return 0;
        });

        var root = new RootCommand("testwinai — Windows AI / Snapdragon X Elite NPU exploration.")
        {
            probeCmd, generateCmd, chatCmd, benchCmd,
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
        Console.WriteLine("    testwinai chat     --model <dir>              multi-turn REPL with KV-cache");
        Console.WriteLine("    testwinai bench    --model <dir>              system-prompt × task suite)");
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

    // ---------- Bench: system-prompt × task suite ----------
    //
    // Cross-product runner: for each (system_prompt, task) cell, prefill a fresh Generator
    // with the system + user wrapped in ChatML, decode to EOS / cap / collapse, capture
    // response + timings + cheap heuristic flags. Writes a structured JSON file (for
    // offline LLM-as-judge scoring) and a markdown file (for eyeballing).

    private sealed record BenchSystemPrompt(string Id, string Text);
    private sealed record BenchTask(string Id, string Prompt);
    private sealed record BenchSuite(BenchSystemPrompt[] SystemPrompts, BenchTask[] Tasks);

    private sealed record BenchHeuristics(
        bool EndsWithQuestion,
        bool RestatedPrompt,
        bool HitTokenCap,
        bool Collapsed);

    private sealed record BenchResult(
        string TaskId,
        string SystemPromptId,
        string TaskPrompt,
        string SystemText,
        string Response,
        int Tokens,
        long PrefillMs,
        long DecodeMs,
        double DecodeTokPerSec,
        string StopReason,
        BenchHeuristics Heuristics);

    private sealed record BenchRunConfig(
        double Temperature,
        double TopP,
        double RepetitionPenalty,
        int MaxTokensPerTurn);

    private sealed record BenchOutput(
        string ModelDir,
        string ModelName,
        string StartedAt,
        BenchRunConfig Config,
        BenchResult[] Results);

    [JsonSourceGenerationOptions(
        PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
        WriteIndented = true)]
    [JsonSerializable(typeof(BenchSuite))]
    [JsonSerializable(typeof(BenchOutput))]
    private partial class BenchJsonContext : JsonSerializerContext { }

    // Built-in default suite — covers the failure modes we hit in earlier sessions
    // (code task that collapsed into whitespace, list task that drifted into Chinese,
    // length-discipline / yes-no compliance, no unsolicited follow-ups).
    private const string DefaultBenchSuite = """
    {
      "systemPrompts": [
        { "id": "current",  "text": "You are a concise assistant. Keep replies focused. Avoid unsolicited follow-up questions and don't restate the prompt." },
        { "id": "minimal",  "text": "You are a helpful assistant." },
        { "id": "strict",   "text": "Answer in as few words as possible. Do not add explanations unless explicitly requested. Do not ask follow-up questions. Never restate the question." },
        { "id": "format",   "text": "You are a concise assistant. Follow the user's requested format exactly. Code goes in fenced blocks. Yes/no questions get a yes or no. Lists get lists. Do not ask follow-ups." }
      ],
      "tasks": [
        { "id": "palindrome",     "prompt": "Write a Python function to check if a string is a palindrome." },
        { "id": "continents",     "prompt": "List the 3 most populous continents and their approximate populations." },
        { "id": "hexagon",        "prompt": "What is the Qualcomm Hexagon NPU? Answer in one sentence." },
        { "id": "sky-yesno",      "prompt": "Is the sky blue? Answer yes or no." },
        { "id": "python-install", "prompt": "How do I install Python on Windows?" },
        { "id": "translate-fr",   "prompt": "Translate 'hello world' to French." },
        { "id": "math",           "prompt": "What is 17 multiplied by 23?" },
        { "id": "hamlet",         "prompt": "Summarize the plot of Hamlet in one sentence." }
      ]
    }
    """;

    private static void RunBench(
        string modelDir,
        string? suitePath,
        string outputDir,
        double temperature,
        double topP,
        double repetitionPenalty,
        int maxTokensPerTurn)
    {
        Console.WriteLine();
        Console.WriteLine("== Bench (Qwen2.5 on QNN HTP) ==");
        Console.WriteLine($"Model dir: {modelDir}");

        if (!Directory.Exists(modelDir))
        {
            Console.WriteLine($"  Model directory does not exist: {modelDir}");
            return;
        }

        string suiteJson;
        if (suitePath is null)
        {
            Console.WriteLine("  Suite: <built-in default>");
            suiteJson = DefaultBenchSuite;
        }
        else
        {
            Console.WriteLine($"  Suite: {suitePath}");
            suiteJson = File.ReadAllText(suitePath);
        }

        BenchSuite suite;
        try
        {
            suite = JsonSerializer.Deserialize(suiteJson, BenchJsonContext.Default.BenchSuite)
                ?? throw new InvalidOperationException("suite deserialized to null");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Failed to parse suite: {ex.GetType().Name}: {ex.Message}");
            return;
        }

        var totalCells = suite.SystemPrompts.Length * suite.Tasks.Length;
        Console.WriteLine(
            $"  System prompts: {suite.SystemPrompts.Length}, tasks: {suite.Tasks.Length}, cells: {totalCells}");
        Console.WriteLine(
            $"  temperature={temperature}, top-p={topP}, rep-penalty={repetitionPenalty}, max-tokens-per-turn={maxTokensPerTurn}");

        var (model, tokenizer, loadMs) = LoadQnnModel(modelDir);
        try
        {
            Console.WriteLine($"  load: {loadMs} ms");

            var results = new List<BenchResult>(totalCells);
            var cellIdx = 0;
            var swAll = Stopwatch.StartNew();

            foreach (var sys in suite.SystemPrompts)
            {
                foreach (var task in suite.Tasks)
                {
                    cellIdx++;
                    Console.WriteLine();
                    Console.WriteLine($"-- [{cellIdx}/{totalCells}] task={task.Id} system={sys.Id} --");

                    var result = RunBenchCell(
                        model, tokenizer, sys, task,
                        temperature, topP, repetitionPenalty, maxTokensPerTurn);
                    results.Add(result);

                    var flags = new List<string>();
                    if (result.Heuristics.EndsWithQuestion) flags.Add("?");
                    if (result.Heuristics.RestatedPrompt) flags.Add("restate");
                    if (result.Heuristics.HitTokenCap) flags.Add("cap");
                    if (result.Heuristics.Collapsed) flags.Add("collapse");
                    var flagStr = flags.Count > 0 ? $" [{string.Join(",", flags)}]" : "";

                    Console.WriteLine(
                        $"  {result.Tokens} tok, prefill={result.PrefillMs} ms, decode={result.DecodeMs} ms " +
                        $"({result.DecodeTokPerSec:0.0} tok/s), stop={result.StopReason}{flagStr}");
                }
            }

            swAll.Stop();
            Console.WriteLine();
            Console.WriteLine($"All cells done in {swAll.ElapsedMilliseconds / 1000.0:0.0} s.");

            Directory.CreateDirectory(outputDir);
            var timestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
            var jsonPath = Path.Combine(outputDir, $"bench-{timestamp}.json");
            var mdPath = Path.Combine(outputDir, $"bench-{timestamp}.md");

            var modelName = Path.GetFileName(
                modelDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            var output = new BenchOutput(
                modelDir,
                modelName,
                DateTime.Now.ToString("o"),
                new BenchRunConfig(temperature, topP, repetitionPenalty, maxTokensPerTurn),
                results.ToArray());

            File.WriteAllText(
                jsonPath,
                JsonSerializer.Serialize(output, BenchJsonContext.Default.BenchOutput));
            File.WriteAllText(mdPath, RenderBenchMarkdown(output, suite));

            Console.WriteLine("Wrote:");
            Console.WriteLine($"  {jsonPath}");
            Console.WriteLine($"  {mdPath}");
        }
        finally
        {
            tokenizer.Dispose();
            model.Dispose();
        }
    }

    private static BenchResult RunBenchCell(
        Model model,
        Tokenizer tokenizer,
        BenchSystemPrompt sys,
        BenchTask task,
        double temperature,
        double topP,
        double repetitionPenalty,
        int maxTokensPerTurn)
    {
        using var gparams = new GeneratorParams(model);
        // Cells are independent; max_length is the session ceiling, not the per-turn cap.
        gparams.SetSearchOption("max_length", 4096);
        gparams.SetSearchOption("do_sample", temperature > 0.0);
        gparams.SetSearchOption("temperature", temperature);
        gparams.SetSearchOption("top_p", topP);
        gparams.SetSearchOption("repetition_penalty", repetitionPenalty);

        using var generator = new Generator(model, gparams);

        if (!string.IsNullOrWhiteSpace(sys.Text))
        {
            using var sysTokens = tokenizer.Encode(ChatMlSystemOpen + sys.Text + ChatMlSystemClose);
            generator.AppendTokenSequences(sysTokens);
        }
        using var userTokens = tokenizer.Encode(
            ChatMlUserOpen + task.Prompt + ChatMlUserClose + ChatMlAssistantOpen);
        generator.AppendTokenSequences(userTokens);

        using var stream = tokenizer.CreateStream();
        var sb = new StringBuilder();
        var sw = Stopwatch.StartNew();
        long firstTokenMs = 0;
        int tokens = 0;
        int consecutiveBlank = 0;
        bool collapsed = false;
        bool capped = false;
        string stopReason = "eos";

        try
        {
            while (!generator.IsDone())
            {
                generator.GenerateNextToken();
                if (tokens == 0) firstTokenMs = sw.ElapsedMilliseconds;

                var seq = generator.GetSequence(0);
                int lastToken = seq[seq.Length - 1];
                if (lastToken == QwenImEndToken) break;

                var decoded = stream.Decode(lastToken);
                sb.Append(decoded);
                tokens++;

                if (string.IsNullOrWhiteSpace(decoded))
                {
                    consecutiveBlank++;
                    if (consecutiveBlank > 16) { collapsed = true; stopReason = "collapse"; break; }
                }
                else
                {
                    consecutiveBlank = 0;
                }

                if (tokens >= maxTokensPerTurn) { capped = true; stopReason = "cap"; break; }
            }
        }
        catch (Exception ex)
        {
            stopReason = $"error:{ex.GetType().Name}";
        }
        sw.Stop();

        var decodeMs = sw.ElapsedMilliseconds - firstTokenMs;
        var tokPerSec = tokens > 1 && decodeMs > 0
            ? (tokens - 1) * 1000.0 / decodeMs
            : 0.0;

        var response = sb.ToString();
        return new BenchResult(
            task.Id, sys.Id, task.Prompt, sys.Text, response,
            tokens, firstTokenMs, decodeMs, tokPerSec, stopReason,
            ComputeHeuristics(task.Prompt, response, capped, collapsed));
    }

    private static BenchHeuristics ComputeHeuristics(
        string prompt, string response, bool capped, bool collapsed)
    {
        var trimmed = response.TrimEnd();
        var endsWithQuestion = trimmed.EndsWith('?');

        // Restated-prompt heuristic: drop non-alphanumerics, lowercase, then check whether
        // the response opens with the same ~30 chars as the prompt. Cheap and cheerful;
        // catches "To check if a string is a palindrome…" echoed back at us, not paraphrases.
        var promptNorm = NormalizeForOverlap(prompt);
        var responseNorm = NormalizeForOverlap(response);
        var compareLen = Math.Min(30, Math.Min(promptNorm.Length, responseNorm.Length));
        var restated = compareLen >= 10 &&
            promptNorm.AsSpan(0, compareLen).SequenceEqual(responseNorm.AsSpan(0, compareLen));

        return new BenchHeuristics(endsWithQuestion, restated, capped, collapsed);

        static string NormalizeForOverlap(string s)
        {
            var sb = new StringBuilder(s.Length);
            foreach (var c in s)
                if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
            return sb.ToString();
        }
    }

    private static string RenderBenchMarkdown(BenchOutput output, BenchSuite suite)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Bench results");
        sb.AppendLine();
        sb.AppendLine($"- Model: `{output.ModelName}`");
        sb.AppendLine($"- Started: {output.StartedAt}");
        sb.AppendLine(
            $"- Config: temperature={output.Config.Temperature}, top-p={output.Config.TopP}, " +
            $"repetition-penalty={output.Config.RepetitionPenalty}, " +
            $"max-tokens-per-turn={output.Config.MaxTokensPerTurn}");
        sb.AppendLine();

        sb.AppendLine("## System prompts");
        sb.AppendLine();
        foreach (var sys in suite.SystemPrompts)
        {
            sb.AppendLine($"### `{sys.Id}`");
            sb.AppendLine($"> {sys.Text}");
            sb.AppendLine();
        }

        sb.AppendLine("## Results by task");
        sb.AppendLine();

        foreach (var task in suite.Tasks)
        {
            sb.AppendLine($"### Task: `{task.Id}`");
            sb.AppendLine();
            sb.AppendLine($"**Prompt:** {task.Prompt}");
            sb.AppendLine();

            foreach (var cell in output.Results)
            {
                if (cell.TaskId != task.Id) continue;

                var flags = new List<string>();
                if (cell.Heuristics.EndsWithQuestion) flags.Add("ends-with-?");
                if (cell.Heuristics.RestatedPrompt) flags.Add("restated");
                if (cell.Heuristics.HitTokenCap) flags.Add("hit-cap");
                if (cell.Heuristics.Collapsed) flags.Add("collapse");
                var flagStr = flags.Count > 0 ? $" — flags: {string.Join(", ", flags)}" : "";

                sb.AppendLine(
                    $"#### `{cell.SystemPromptId}` " +
                    $"({cell.Tokens} tok, {cell.DecodeTokPerSec:0.0} tok/s, " +
                    $"stop={cell.StopReason}{flagStr})");
                sb.AppendLine();
                sb.AppendLine("```");
                sb.AppendLine(cell.Response.Trim());
                sb.AppendLine("```");
                sb.AppendLine();
            }
        }

        return sb.ToString();
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
