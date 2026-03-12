# Retool: RL Training with AWS AgentCore Sandbox

Slime-based RL training for math problem solving using CodeSandboxEnv (based on AWS AgentCore CodeInterpreter) as a Python execution environment. Training models to leverage code tools to solve math problems.

## Files

- **`generate_with_code_sandbox.py`**: Core generation and reward computation logic
  - `generate_and_rm()`: Async function that generates responses and computes rewards
  - `rollout_logging_with_tool_stats()`: Custom logging function for tool usage metrics

- **`run_retool_qwen3_8b.sh`**: Training script for Qwen3-8B model
## Quick Start

### Prerequisites

1. **Docker**: We recommend to use docker images from the official repo `slimerl/slime`. This example is tested on `slimerl/slime:v0.2.2`

2. **Install strands-env**:
   ```bash
   git clone https://github.com/horizon-rl/strands-env.git
   cd strands-env
   pip install -e .
   ```

3. **Data**: Prepare training and validation data. Following retool, we use `DAPO-Math-17k` and `AIME-2024` as training and evaluation data.  We directly use the data copies from slime: `zhuzilin/dapo-math-17k` and `zhuzilin/aime-2024`.
   - Training: `/root/data/dapo-math-17k.jsonl`
   - Validation: `/root/data/aime-2024.jsonl`

4. **Model checkpoints**:
   - HuggingFace: `/root/Qwen3-8B`
   - Megatron-Core: `/root/Qwen3-8B-mcore` (convert using [slime guide](https://github.com/THUDM/slime/blob/e70e6476632ebd551d5eb82186ad1cc948e0d8f5/docs/en/get_started/quick_start.md?plain=1#L70))

### Run Training

```bash
cd strands-env
export WANDB_API_KEY="YOUR_KEY"
bash examples/slime/retool/run_retool_qwen3_8b.sh
```

### Reward

The reward combines correctness verification and tool usage incentives:
1. **Correctness check**: MathVerifyReward verifies the answer against ground truth
2. **Tool usage bonus**: Partial credit given for using tools even when incorrect
   - Correct: `reward = 1.0`
   - Incorrect: `reward = min(-0.6, -1 + (tool_iters - 2) / 2 * 0.1)`

## Metrics

Training logs track:
- `rollout/avg_tool_iters`: Mean tool iterations per sample
- `rollout/avg_tool_calls`: Mean total tool calls per sample
- `rollout/tool_usage_ratio`: Percentage of samples using tools

## References
- [slime Framework](https://github.com/THUDM/slime)
- [Retool Paper](https://arxiv.org/pdf/2504.11536)
