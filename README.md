# pi-nvidia-nim

NVIDIA NIM API provider extension for [pi coding agent](https://github.com/earendil-works/pi-mono) - access 100+ models from [build.nvidia.com](https://build.nvidia.com) including Inkling, MiniMax M3, DeepSeek V4 Flash/Pro, DeepSeek V3.2, Kimi K2.6, GLM-5, GLM-4.7, Qwen3, Llama 4, and many more.

https://github.com/user-attachments/assets/f44773e4-9bf8-4bb5-a9c0-d5938030701c


## Setup

### 1. Get an NVIDIA NIM API Key

1. Go to [build.nvidia.com](https://build.nvidia.com)
2. Sign in or create an account
3. Navigate to any model page and click "Get API Key"
4. Copy your key (starts with `nvapi-`)

### 2. Set Your API Key

```bash
# Preferred by this extension
export NVIDIA_NIM_API_KEY=nvapi-your-key-here

# Also supported, matching NVIDIA's website examples
export NVIDIA_API_KEY=nvapi-your-key-here
```

Add one of these to your `~/.bashrc`, `~/.zshrc`, or shell profile to persist it.

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
pi --provider nvidia-nim --model "deepseek-ai/deepseek-v4-flash"

# With thinking enabled
pi --provider nvidia-nim --model "deepseek-ai/deepseek-v4-flash" --thinking high

# Limit model cycling to NIM models
pi --models "nvidia-nim/*"
```

## Reasoning / Thinking

NVIDIA NIM models use a non-standard `chat_template_kwargs` parameter to enable thinking, rather than the standard OpenAI `reasoning_effort`. This extension handles this automatically via a custom streaming wrapper that injects the correct per-model parameters.

### How it works

When you change the thinking level in pi (`Shift+Tab` to cycle), the extension:

1. **Maps thinking levels** to values each NIM model accepts. MiniMax M3 uses its native `disabled`/`adaptive`/`enabled` modes; Inkling and DeepSeek V4 map extended levels to `max`.
2. **Injects `chat_template_kwargs`** per model to actually enable thinking:
   - Inkling: `{ reasoning_effort: "none" | "minimal" | "low" | "medium" | "high" | "max" }`
   - MiniMax M3: `{ thinking_mode: "disabled" | "adaptive" | "enabled" }`
   - DeepSeek V4: `{ thinking: true, reasoning_effort: "high" | "max" }`
   - DeepSeek V3.x, R1 distills: `{ thinking: true }`
   - GLM-5, GLM-4.7: `{ enable_thinking: true, clear_thinking: false }`
   - Kimi K2.6, K2-thinking: `{ thinking: true }`
   - Qwen3, QwQ: `{ enable_thinking: true }`
3. **Explicitly selects the lowest/off setting** when thinking is "off" for configurable models. MiniMax M3 uses `disabled`; Inkling uses `none` as a conditioning hint rather than a guarantee of zero reasoning tokens.
4. **Uses `system` role** instead of `developer` for all NIM models - the `developer` role combined with `chat_template_kwargs` causes 500 errors on NIM.

### Supported thinking levels

| pi Level | NIM Mapping | Effect |
|----------|-------------|--------|
| off | MiniMax M3 `disabled`; Inkling `none`; explicit disable elsewhere | No/lowest reasoning effort |
| minimal | MiniMax M3 `adaptive`; Inkling `minimal`; low/high where required | Thinking enabled when useful |
| low | MiniMax M3 `adaptive`; low, or high for DeepSeek V4 | Thinking enabled when useful |
| medium | MiniMax M3 `adaptive`; medium, or high for DeepSeek V4 | Thinking enabled when useful |
| high | MiniMax M3 `enabled`; high elsewhere | Thinking enabled |
| xhigh | MiniMax M3 `enabled`; max for Inkling/DeepSeek V4 | Extended thinking |
| max | MiniMax M3 `enabled`; max for Inkling/DeepSeek V4 | Maximum supported thinking |

## Available Models

The extension ships with curated metadata for 44 featured models. At startup, it also queries the NVIDIA NIM API to discover additional models automatically.

### Featured Models

| Model | Reasoning | Vision | Context |
|-------|-----------|--------|---------|
| `thinkingmachines/inkling` | ✅ | ✅ | 1M |
| `minimaxai/minimax-m3` | ✅ | ✅ | 1M |
| `deepseek-ai/deepseek-v4-flash` | ✅ | | 1M |
| `deepseek-ai/deepseek-v4-pro` | ✅ | | 1M |
| `deepseek-ai/deepseek-v3.2` | ✅ | | 128K |
| `deepseek-ai/deepseek-v3.1` | ✅ | | 128K |
| `moonshotai/kimi-k2.6` | ✅ | | 256K |
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

All major models support OpenAI-compatible tool calling. Tested and confirmed working with DeepSeek V4/V3.2, GLM-5, GLM-4.7, Qwen3, Kimi K2.6, and others.

## How It Works

This extension uses `pi.registerProvider()` to register NVIDIA NIM as a custom provider with a custom `streamSimple` wrapper around pi's built-in `openai-completions` streamer.

The custom streamer:
1. Intercepts the request payload via `onPayload` callback
2. Injects `chat_template_kwargs` for models that need it to enable thinking
3. Maps unsupported thinking levels to NIM-compatible values while preserving MiniMax M3's native modes and Inkling's native effort presets
4. Suppresses `reasoning_effort` for models that don't respond to it (e.g., DeepSeek without kwargs)
5. Uses the standard OpenAI SSE streaming format - pi already parses `reasoning_content` and `reasoning` fields from streaming deltas

## Configuration

The only configuration needed is either the `NVIDIA_NIM_API_KEY` or `NVIDIA_API_KEY` environment variable. All models on NVIDIA NIM are free during the preview period (with rate limits).

### Disable dynamic model discovery

By default the extension queries the NIM `/models` endpoint on session start and registers any additional chat models it finds beyond the curated list. To keep only the curated models, set:

```bash
export NVIDIA_NIM_DISABLE_DYNAMIC_DISCOVERY=true
```

## Notes

- All costs are set to `$0` since NVIDIA NIM preview models are free (rate-limited)
- Context windows and max tokens are best-effort estimates; some may differ from actual API limits
- If a model isn't in the curated list, it gets a conservative 32K context window and 8K max output tokens
- The extension filters out embedding, reward, safety, and other non-chat models automatically
- Rate limits on free preview keys are relatively strict; you may encounter 429 errors during heavy usage
- MiniMax M3 streams thinking through `reasoning_content`; older MiniMax models may use `<think>` tags inline in content

## License

MIT
