Write or update a README.md for an eval benchmark directory.

The user provides the eval directory path as $ARGUMENTS (e.g., `examples/eval/aime`). If not provided, ask.

## Steps

1. **Read all files** in the eval directory — `.py`, `.md`, `.txt`, etc. Understand what the benchmark does, which environment it uses, and how to run it.

2. **Identify the key details:**
   - Benchmark name and one-line description
   - Which environment class is used (e.g., `AgentCoreCodeEnv`, `CalculatorEnv`)
   - Prerequisites beyond the base install (Docker, extra pip packages, AWS credentials, etc.)
   - What files are in the directory and what each does
   - The CLI command(s) to run the eval

3. **Write `README.md`** following this exact template:

```markdown
# {Benchmark Name}

{One-line description of what this benchmark evaluates.}

## Variants

{Table of registered benchmark names and descriptions. Only include this section if the benchmark has multiple registered variants or a registered name.}

| Name | Description |
|---|---|
| `variant-1` | ... |
| `variant-2` | ... |

## Setup

{Prerequisites. If none beyond base install, write: "No additional setup required beyond `strands-env`."}

## Environments

{Table (File | Description) of the env hook modules — each row is an environment option mapping to the `--env` flag (e.g. chat-only vs tools).}

## Usage

{CLI command(s) in a bash code block. Use paths relative to the repo root. Show the registered benchmark name if available. Include key arguments that users will commonly adjust: `--base-url`, `--max-tokens`, `--n-samples-per-prompt`, `--max-concurrency`, and `--max-tool-iters` if relevant. Omit arguments with sensible defaults that rarely need changing (e.g., `--temperature`, `-o` output folder).}

See `python -m strands_env.eval --help` for all CLI options.
```

## Style rules

- Use `-` (hyphen) as separator in lists (e.g., `**file.py** - Description`)
- Keep descriptions concise — one sentence per item
- CLI paths should be relative to the repo root
- Do NOT include license headers
- Do NOT include sections that have no content (e.g., skip Setup if truly nothing needed)
- If the benchmark has a registered name (e.g., `aime-2024`), show it in the CLI command
- Only expose CLI args that users commonly adjust; omit those with sensible defaults

## Reference

Use these existing eval READMEs as style references:

- `examples/eval/aime/README.md`
- `examples/eval/simple_math/README.md`
- `examples/eval/terminal_bench/README.md`

Read them first to match the current conventions, then apply the template above to improve consistency across all of them.
