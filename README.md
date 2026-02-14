# pi-nvidia-nim

NVIDIA NIM API provider extension for [pi coding agent](https://github.com/badlogic/pi-mono) - access 100+ models from [build.nvidia.com](https://build.nvidia.com) including DeepSeek V3.2, Kimi K2.5, MiniMax M2.1, GLM-5, GLM-4.7, Qwen3, Llama 4, and many more.

https://github.com/user-attachments/assets/f44773e4-9bf8-4bb5-a9c0-d5938030701c


## Setup

### 1. Get an NVIDIA NIM API Key

1. Go to [build.nvidia.com](https://build.nvidia.com)
2. Sign in or create an account
3. Navigate to any model page and click "Get API Key"
4. Copy your key (starts with `nvapi-`)

### 2. Set Your API Key

```bash
export NVIDIA_NIM_API_KEY=nvapi-your-key-here
```

Add this to your `~/.bashrc`, `~/.zshrc`, or shell profile to persist it.

### 3. Install the Extension

**As a pi package (recommended):**

```bash
pi install git:github.com/xRyul/pi-nvidia-nim
```

**Or load directly:**

```bash
pi -e /path/to/pi-nvidia-nim
```

**Or copy to your extensions directory:**

```bash
cp -r pi-nvidia-nim ~/.pi/agent/extensions/pi-nvidia-nim
```

## Usage

Once loaded, NVIDIA NIM models appear in the `/model` selector under the `nvidia-nim` provider. You can also:

- Press **Ctrl+L** to open the model selector and search for `nvidia-nim`
- Use `/scoped-models` to pin your favourite NIM models for quick switching

### CLI

```bash
# Use a specific NIM model directly
pi --provider nvidia-nim --model "deepseek-ai/deepseek-v3.2"

# With thinking enabled
pi --provider nvidia-nim --model "deepseek-ai/deepseek-v3.2" --thinking low

# Limit model cycling to NIM models
pi --models "nvidia-nim/*"
```

## Reasoning / Thinking

NVIDIA NIM models use a non-standard `chat_template_kwargs` parameter to enable thinking, rather than the standard OpenAI `reasoning_effort`. This extension handles this automatically via a custom streaming wrapper that injects the correct per-model parameters.

### How it works

When you change the thinking level in pi (`Shift+Tab` to cycle), the extension:

1. **Maps `"minimal"` → `"low"`** - NIM only accepts `low`, `medium`, `high` (not `minimal`). Selecting "minimal" in pi works fine; it's silently mapped.
2. **Injects `chat_template_kwargs`** per model to actually enable thinking:
   - DeepSeek V3.x, R1 distills: `{ thinking: true }`
   - GLM-5, GLM-4.7: `{ enable_thinking: true, clear_thinking: false }`
   - Kimi K2.5, K2-thinking: `{ thinking: true }`
   - Qwen3, QwQ: `{ enable_thinking: true }`
3. **Explicitly disables thinking** when the level is "off" for models that think by default (e.g., GLM-5, GLM-4.7).
4. **Uses `system` role** instead of `developer` for all NIM models - the `developer` role combined with `chat_template_kwargs` causes 500 errors on NIM.

### Supported thinking levels

| pi Level | NIM Mapping | Effect |
|----------|-------------|--------|
| off | No kwargs (or explicit disable) | No reasoning output |
| minimal | Mapped to "low" | Thinking enabled |
| low | low | Thinking enabled |
| medium | medium | Thinking enabled |
| high | high | Thinking enabled |

## Available Models

The extension ships with curated metadata for 39 featured models. At startup, it also queries the NVIDIA NIM API to discover additional models automatically.

### Featured Models

| Model | Reasoning | Vision | Context |
|-------|-----------|--------|---------|
| `deepseek-ai/deepseek-v3.2` | ✅ | | 128K |
| `deepseek-ai/deepseek-v3.1` | ✅ | | 128K |
| `moonshotai/kimi-k2.5` | ✅ | | 256K |
| `moonshotai/kimi-k2-thinking` | ✅ | | 128K |
| `minimaxai/minimax-m2.1` | | | 1M |
| `z-ai/glm5` | ✅ | | 128K |
| `z-ai/glm4.7` | ✅ | | 128K |
| `openai/gpt-oss-120b` | | | 128K |
| `qwen/qwen3-coder-480b-a35b-instruct` | ✅ | | 256K |
| `qwen/qwen3-235b-a22b` | ✅ | | 128K |
| `meta/llama-4-maverick-17b-128e-instruct` | | | 1M |
| `meta/llama-3.1-405b-instruct` | | | 128K |
| `meta/llama-3.2-90b-vision-instruct` | | ✅ | 128K |
| `mistralai/mistral-large-3-675b-instruct-2512` | | | 128K |
| `mistralai/devstral-2-123b-instruct-2512` | | | 128K |
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | ✅ | | 128K |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | ✅ | | 128K |
| `microsoft/phi-4-mini-flash-reasoning` | ✅ | | 128K |
| `ibm/granite-3.3-8b-instruct` | | | 128K |

...and 20+ more curated models, plus automatic discovery of new models from the API.

### Tool Calling

All major models support OpenAI-compatible tool calling. Tested and confirmed working with DeepSeek V3.2, GLM-5, GLM-4.7, Qwen3, Kimi K2.5, and others.

## How It Works

This extension uses `pi.registerProvider()` to register NVIDIA NIM as a custom provider with a custom `streamSimple` wrapper around pi's built-in `openai-completions` streamer.

The custom streamer:
1. Intercepts the request payload via `onPayload` callback
2. Injects `chat_template_kwargs` for models that need it to enable thinking
3. Maps unsupported thinking levels (`minimal` → `low`)
4. Suppresses `reasoning_effort` for models that don't respond to it (e.g., DeepSeek without kwargs)
5. Uses the standard OpenAI SSE streaming format - pi already parses `reasoning_content` and `reasoning` fields from streaming deltas

## Configuration

The only configuration needed is the `NVIDIA_NIM_API_KEY` environment variable. All models on NVIDIA NIM are free during the preview period (with rate limits).

## Notes

- All costs are set to `$0` since NVIDIA NIM preview models are free (rate-limited)
- Context windows and max tokens are best-effort estimates; some may differ from actual API limits
- If a model isn't in the curated list, it gets a conservative 32K context window and 8K max output tokens
- The extension filters out embedding, reward, safety, and other non-chat models automatically
- Rate limits on free preview keys are relatively strict; you may encounter 429 errors during heavy usage
- MiniMax models use `<think>` tags inline in content rather than the `reasoning_content` field

## License

MIT
