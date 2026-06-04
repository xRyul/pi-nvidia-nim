/**
 * NVIDIA NIM API Provider Extension for pi
 *
 * Provides access to 100+ models from NVIDIA's NIM platform (build.nvidia.com)
 * via their OpenAI-compatible API endpoint.
 *
 * Setup:
 *   1. Get an API key from https://build.nvidia.com
 *   2. Export it: export NVIDIA_NIM_API_KEY=nvapi-... (or NVIDIA_API_KEY=nvapi-...)
 *   3. Load the extension:
 *      pi -e ./path/to/pi-nvidia-nim
 *      # or install as a package:
 *      pi install git:github.com/user/pi-nvidia-nim
 *
 * Then use /model and search for "nvidia-nim/" to see all available models.
 *
 * ## Reasoning / Thinking
 *
 * NVIDIA NIM models use `chat_template_kwargs` to enable thinking, which differs
 * from the standard OpenAI `reasoning_effort` parameter. This extension wraps the
 * standard streaming implementation and injects the correct per-model thinking
 * parameters:
 *
 * - DeepSeek V3.x: `chat_template_kwargs: { thinking: true }`
 * - DeepSeek V4:   `chat_template_kwargs: { thinking: true, reasoning_effort: "high" | "max" }`
 * - GLM-5.1/5/4.7: `chat_template_kwargs: { enable_thinking: true, clear_thinking: false }`
 * - Kimi K2.5:     `chat_template_kwargs: { thinking: true }` (also accepts reasoning_effort)
 * - Qwen3:         `chat_template_kwargs: { enable_thinking: true }`
 *
 * NIM only accepts selected `reasoning_effort` values. The extension maps pi's
 * provider-agnostic levels to the values each NIM model accepts.
 *
 * Some models (e.g., GLM-5.1, GLM-5, GLM-4.7) always produce reasoning output regardless of
 * thinking settings.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type {
	Api,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { streamSimpleOpenAICompletions } from "@earendil-works/pi-ai";
import { getAgentDir, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

// =============================================================================
// Constants
// =============================================================================

const NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1";
const NVIDIA_NIM_API_KEY_ENV = "NVIDIA_NIM_API_KEY";
const NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY";
const NVIDIA_API_KEY_ENV_NAMES = [NVIDIA_NIM_API_KEY_ENV, NVIDIA_API_KEY_ENV] as const;
const PROVIDER_NAME = "nvidia-nim";

// =============================================================================
// Per-model thinking configuration
// =============================================================================

/**
 * Maps model ID prefixes/exact IDs to their chat_template_kwargs for thinking.
 * When a user enables thinking in pi (any level > off), we inject these kwargs
 * into the request body. Models not listed here either:
 * - Don't support thinking (non-reasoning models)
 * - Always think regardless (GLM models without explicit kwargs)
 * - Work with standard reasoning_effort (rare on NIM)
 */
interface ThinkingConfig {
	/** chat_template_kwargs to send when thinking is enabled */
	enableKwargs: Record<string, unknown>;
	/** chat_template_kwargs to send when thinking is explicitly disabled (optional) */
	disableKwargs?: Record<string, unknown>;
	/** If true, also send reasoning_effort alongside chat_template_kwargs */
	sendReasoningEffort?: boolean;
	/** If true, include a model-specific reasoning_effort inside chat_template_kwargs */
	includeReasoningEffortInKwargs?: boolean;
}

const THINKING_CONFIGS: Record<string, ThinkingConfig> = {
	// DeepSeek models need chat_template_kwargs - reasoning_effort alone doesn't trigger thinking
	"deepseek-ai/deepseek-v4-flash": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
		includeReasoningEffortInKwargs: true,
	},
	"deepseek-ai/deepseek-v4-pro": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
		includeReasoningEffortInKwargs: true,
	},
	"deepseek-ai/deepseek-v3.2": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-v3.1": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-v3.1-terminus": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-r1-distill-llama-8b": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-r1-distill-qwen-7b": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-r1-distill-qwen-14b": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"deepseek-ai/deepseek-r1-distill-qwen-32b": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	// GLM models (Z-AI) - think by default, but can be controlled
	"z-ai/glm4.7": {
		enableKwargs: { enable_thinking: true, clear_thinking: false },
		disableKwargs: { enable_thinking: false },
	},
	"z-ai/glm5": {
		enableKwargs: { enable_thinking: true, clear_thinking: false },
		disableKwargs: { enable_thinking: false },
	},
	"z-ai/glm5.1": {
		enableKwargs: { enable_thinking: true, clear_thinking: false },
		disableKwargs: { enable_thinking: false },
	},
	// Kimi models: chat_template_kwargs works, reasoning_effort also works
	"moonshotai/kimi-k2.6": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
		sendReasoningEffort: true,
	},
	"moonshotai/kimi-k2-thinking": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
		sendReasoningEffort: true,
	},
	// Qwen3 reasoning models
	"qwen/qwen3-235b-a22b": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	"qwen/qwen3-coder-480b-a35b-instruct": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	"qwen/qwen3-next-80b-a3b-thinking": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	"qwen/qwq-32b": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	// Microsoft Phi reasoning
	"microsoft/phi-4-mini-flash-reasoning": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
	// NVIDIA Nemotron reasoning models
	"nvidia/llama-3.1-nemotron-ultra-253b-v1": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"nvidia/llama-3.3-nemotron-super-49b-v1": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	"nvidia/llama-3.3-nemotron-super-49b-v1.5": {
		enableKwargs: { thinking: true },
		disableKwargs: { thinking: false },
	},
	// Mistral reasoning
	"mistralai/magistral-small-2506": {
		enableKwargs: { enable_thinking: true },
		disableKwargs: { enable_thinking: false },
	},
};

// =============================================================================
// Reasoning models and their capabilities
// =============================================================================

const REASONING_MODELS = new Set(Object.keys(THINKING_CONFIGS));

// Models known to support image/vision input
const VISION_MODELS = new Set([
	"meta/llama-3.2-11b-vision-instruct",
	"meta/llama-3.2-90b-vision-instruct",
	"microsoft/phi-3-vision-128k-instruct",
	"microsoft/phi-3.5-vision-instruct",
	"microsoft/phi-4-multimodal-instruct",
	"nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
	"nvidia/nemotron-nano-12b-v2-vl",
	"nvidia/cosmos-reason2-8b",
]);

// Embedding / non-chat models to skip
const SKIP_MODELS = new Set([
	"baai/bge-m3",
	"nvidia/embed-qa-4",
	"nvidia/nv-embed-v1",
	"nvidia/nv-embedcode-7b-v1",
	"nvidia/nv-embedqa-e5-v5",
	"nvidia/nv-embedqa-mistral-7b-v2",
	"nvidia/nvclip",
	"nvidia/streampetr",
	"nvidia/vila",
	"nvidia/neva-22b",
	"nvidia/nemoretriever-parse",
	"nvidia/nemotron-parse",
	"nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
	"nvidia/llama-3.2-nemoretriever-300m-embed-v1",
	"nvidia/llama-3.2-nemoretriever-300m-embed-v2",
	"nvidia/llama-3.2-nv-embedqa-1b-v1",
	"nvidia/llama-3.2-nv-embedqa-1b-v2",
	"nvidia/llama-nemotron-embed-vl-1b-v2",
	"nvidia/llama-3.1-nemotron-70b-reward",
	"nvidia/nemotron-4-340b-reward",
	"nvidia/nemotron-content-safety-reasoning-4b",
	"nvidia/llama-3.1-nemoguard-8b-content-safety",
	"nvidia/llama-3.1-nemoguard-8b-topic-control",
	"nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
	"meta/llama-guard-4-12b",
	"nvidia/riva-translate-4b-instruct",
	"nvidia/riva-translate-4b-instruct-v1.1",
	"google/deplot",
	"google/paligemma",
	"google/recurrentgemma-2b",
	"google/shieldgemma-9b",
	"microsoft/kosmos-2",
	"adept/fuyu-8b",
	"bigcode/starcoder2-15b",
	"bigcode/starcoder2-7b",
	"snowflake/arctic-embed-l",
	"mistralai/mamba-codestral-7b-v0.1",
	"mistralai/mathstral-7b-v0.1",
	"mistralai/mixtral-8x22b-v0.1",
	"nvidia/mistral-nemo-minitron-8b-base",
	"google/gemma-2b",
	"google/gemma-7b",
	"google/codegemma-7b",
	"meta/llama2-70b",
]);

// Known context windows (tokens)
const CONTEXT_WINDOWS: Record<string, number> = {
	// DeepSeek
	"deepseek-ai/deepseek-v3.1": 131072,
	"deepseek-ai/deepseek-v3.1-terminus": 131072,
	"deepseek-ai/deepseek-v3.2": 131072,
	"deepseek-ai/deepseek-v4-flash": 1048576,
	"deepseek-ai/deepseek-v4-pro": 1048576,
	"deepseek-ai/deepseek-r1-distill-llama-8b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-14b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-32b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-7b": 131072,
	"deepseek-ai/deepseek-coder-6.7b-instruct": 16384,
	// Kimi / Moonshot
	"moonshotai/kimi-k2-instruct": 131072,
	"moonshotai/kimi-k2-instruct-0905": 131072,
	"moonshotai/kimi-k2-thinking": 131072,
	"moonshotai/kimi-k2.6": 262144,
	// MiniMax
	"minimaxai/minimax-m2": 1048576,
	"minimaxai/minimax-m2.1": 1048576,
	"minimaxai/minimax-m2.7": 204800,
	// Meta Llama
	"meta/llama-3.1-405b-instruct": 131072,
	"meta/llama-3.1-70b-instruct": 131072,
	"meta/llama-3.1-8b-instruct": 131072,
	"meta/llama-3.2-11b-vision-instruct": 131072,
	"meta/llama-3.2-1b-instruct": 131072,
	"meta/llama-3.2-3b-instruct": 131072,
	"meta/llama-3.2-90b-vision-instruct": 131072,
	"meta/llama-3.3-70b-instruct": 131072,
	"meta/llama-4-maverick-17b-128e-instruct": 1048576,
	"meta/llama-4-scout-17b-16e-instruct": 524288,
	"meta/llama3-70b-instruct": 8192,
	"meta/llama3-8b-instruct": 8192,
	// Mistral
	"mistralai/mistral-large-3-675b-instruct-2512": 131072,
	"mistralai/mistral-medium-3-instruct": 131072,
	"mistralai/devstral-2-123b-instruct-2512": 131072,
	"mistralai/magistral-small-2506": 131072,
	"mistralai/mistral-large": 131072,
	"mistralai/mistral-large-2-instruct": 131072,
	"mistralai/mistral-small-24b-instruct": 32768,
	"mistralai/mistral-small-3.1-24b-instruct-2503": 131072,
	"mistralai/mistral-nemotron": 131072,
	"mistralai/mixtral-8x22b-instruct-v0.1": 65536,
	"mistralai/mixtral-8x7b-instruct-v0.1": 32768,
	"mistralai/codestral-22b-instruct-v0.1": 32768,
	"mistralai/ministral-14b-instruct-2512": 131072,
	// Microsoft Phi
	"microsoft/phi-3-medium-128k-instruct": 131072,
	"microsoft/phi-3-mini-128k-instruct": 131072,
	"microsoft/phi-3-small-128k-instruct": 131072,
	"microsoft/phi-3-medium-4k-instruct": 4096,
	"microsoft/phi-3-mini-4k-instruct": 4096,
	"microsoft/phi-3-small-8k-instruct": 8192,
	"microsoft/phi-3-vision-128k-instruct": 131072,
	"microsoft/phi-3.5-mini-instruct": 131072,
	"microsoft/phi-3.5-moe-instruct": 131072,
	"microsoft/phi-3.5-vision-instruct": 131072,
	"microsoft/phi-4-mini-instruct": 131072,
	"microsoft/phi-4-mini-flash-reasoning": 131072,
	"microsoft/phi-4-multimodal-instruct": 131072,
	// Qwen
	"qwen/qwen2-7b-instruct": 131072,
	"qwen/qwen2.5-7b-instruct": 131072,
	"qwen/qwen2.5-coder-32b-instruct": 131072,
	"qwen/qwen2.5-coder-7b-instruct": 131072,
	"qwen/qwen3-235b-a22b": 131072,
	"qwen/qwen3-coder-480b-a35b-instruct": 262144,
	"qwen/qwen3-next-80b-a3b-instruct": 131072,
	"qwen/qwen3-next-80b-a3b-thinking": 131072,
	"qwen/qwq-32b": 131072,
	// Google Gemma
	"google/gemma-2-27b-it": 8192,
	"google/gemma-2-2b-it": 8192,
	"google/gemma-2-9b-it": 8192,
	"google/gemma-3-12b-it": 131072,
	"google/gemma-3-1b-it": 32768,
	"google/gemma-3-27b-it": 131072,
	"google/gemma-3-4b-it": 131072,
	"google/gemma-3n-e2b-it": 131072,
	"google/gemma-3n-e4b-it": 131072,
	"google/codegemma-1.1-7b": 8192,
	// NVIDIA
	"nvidia/llama-3.1-nemotron-ultra-253b-v1": 131072,
	"nvidia/llama-3.1-nemotron-70b-instruct": 131072,
	"nvidia/llama-3.1-nemotron-51b-instruct": 131072,
	"nvidia/llama-3.3-nemotron-super-49b-v1": 131072,
	"nvidia/llama-3.3-nemotron-super-49b-v1.5": 131072,
	"nvidia/nemotron-4-340b-instruct": 4096,
	"nvidia/nvidia-nemotron-nano-9b-v2": 131072,
	// OpenAI open-source
	"openai/gpt-oss-120b": 131072,
	"openai/gpt-oss-20b": 131072,
	// Z-AI / GLM
	"z-ai/glm4.7": 131072,
	"z-ai/glm5": 131072,
	"z-ai/glm5.1": 131072,
	// StepFun
	"stepfun-ai/step-3.5-flash": 131072,
	// ByteDance
	"bytedance/seed-oss-36b-instruct": 131072,
	// IBM Granite
	"ibm/granite-3.3-8b-instruct": 131072,
	"ibm/granite-3.0-8b-instruct": 8192,
	"ibm/granite-3.0-3b-a800m-instruct": 8192,
	"ibm/granite-34b-code-instruct": 8192,
	"ibm/granite-8b-code-instruct": 8192,
	// Older / smaller models with limited context
	"upstage/solar-10.7b-instruct": 4096,
	"01-ai/yi-large": 32768,
	"databricks/dbrx-instruct": 32768,
	"baichuan-inc/baichuan2-13b-chat": 4096,
	"thudm/chatglm3-6b": 8192,
	"tiiuae/falcon3-7b-instruct": 8192,
	"zyphra/zamba2-7b-instruct": 4096,
	"aisingapore/sea-lion-7b-instruct": 4096,
	"mediatek/breeze-7b-instruct": 4096,
	"meta/codellama-70b": 16384,
	"mistralai/mistral-7b-instruct-v0.2": 32768,
	"mistralai/mistral-7b-instruct-v0.3": 32768,
	"nv-mistralai/mistral-nemo-12b-instruct": 131072,
	"nvidia/nemotron-mini-4b-instruct": 4096,
	"nvidia/nemotron-4-mini-hindi-4b-instruct": 4096,
	"nvidia/usdcode-llama-3.1-70b-instruct": 131072,
	"sarvamai/sarvam-m": 32768,
	"writer/palmyra-creative-122b": 32768,
	"writer/palmyra-fin-70b-32k": 32768,
	"writer/palmyra-med-70b": 8192,
	"writer/palmyra-med-70b-32k": 32768,
	"igenius/colosseum_355b_instruct_16k": 16384,
	"igenius/italia_10b_instruct_16k": 16384,
	"rakuten/rakutenai-7b-chat": 4096,
	"rakuten/rakutenai-7b-instruct": 4096,
};

// Known max output tokens
const MAX_TOKENS: Record<string, number> = {
	"deepseek-ai/deepseek-v3.1": 16384,
	"deepseek-ai/deepseek-v3.1-terminus": 16384,
	"deepseek-ai/deepseek-v3.2": 16384,
	"deepseek-ai/deepseek-v4-flash": 16384,
	"deepseek-ai/deepseek-v4-pro": 16384,
	"moonshotai/kimi-k2.6": 16384,
	"moonshotai/kimi-k2-instruct": 8192,
	"moonshotai/kimi-k2-thinking": 16384,
	"minimaxai/minimax-m2": 8192,
	"minimaxai/minimax-m2.1": 8192,
	"minimaxai/minimax-m2.7": 8192,
	"meta/llama-4-maverick-17b-128e-instruct": 16384,
	"meta/llama-4-scout-17b-16e-instruct": 16384,
	"z-ai/glm4.7": 16384,
	"z-ai/glm5": 16384,
	"z-ai/glm5.1": 16384,
	"qwen/qwen3-coder-480b-a35b-instruct": 65536,
	"nvidia/llama-3.1-nemotron-ultra-253b-v1": 32768,
	"openai/gpt-oss-120b": 16384,
	"openai/gpt-oss-20b": 16384,
	"mistralai/mistral-large-3-675b-instruct-2512": 16384,
	"mistralai/devstral-2-123b-instruct-2512": 32768,
};

// =============================================================================
// Curated "featured" models - listed first in the model selector
// =============================================================================

const FEATURED_MODELS = [
	// Flagship / frontier
	"deepseek-ai/deepseek-v4-flash",
	"deepseek-ai/deepseek-v4-pro",
	"deepseek-ai/deepseek-v3.2",
	"deepseek-ai/deepseek-v3.1",
	"deepseek-ai/deepseek-v3.1-terminus",
	"moonshotai/kimi-k2.6",
	"moonshotai/kimi-k2-thinking",
	"moonshotai/kimi-k2-instruct",
	"moonshotai/kimi-k2-instruct-0905",
	"minimaxai/minimax-m2.1",
	"minimaxai/minimax-m2",
	"minimaxai/minimax-m2.7",
	"z-ai/glm5.1",
	"z-ai/glm5",
	"z-ai/glm4.7",
	"openai/gpt-oss-120b",
	"openai/gpt-oss-20b",
	"stepfun-ai/step-3.5-flash",
	"bytedance/seed-oss-36b-instruct",
	// Qwen
	"qwen/qwen3-coder-480b-a35b-instruct",
	"qwen/qwen3-235b-a22b",
	"qwen/qwen3-next-80b-a3b-instruct",
	"qwen/qwen3-next-80b-a3b-thinking",
	"qwen/qwq-32b",
	"qwen/qwen2.5-coder-32b-instruct",
	// Meta Llama
	"meta/llama-4-maverick-17b-128e-instruct",
	"meta/llama-4-scout-17b-16e-instruct",
	"meta/llama-3.3-70b-instruct",
	"meta/llama-3.1-405b-instruct",
	"meta/llama-3.2-90b-vision-instruct",
	// Mistral
	"mistralai/mistral-large-3-675b-instruct-2512",
	"mistralai/mistral-medium-3-instruct",
	"mistralai/devstral-2-123b-instruct-2512",
	"mistralai/magistral-small-2506",
	"mistralai/mistral-nemotron",
	// NVIDIA
	"nvidia/llama-3.1-nemotron-ultra-253b-v1",
	"nvidia/llama-3.3-nemotron-super-49b-v1.5",
	"nvidia/llama-3.3-nemotron-super-49b-v1",
	// DeepSeek R1 distilled
	"deepseek-ai/deepseek-r1-distill-qwen-32b",
	"deepseek-ai/deepseek-r1-distill-qwen-14b",
	// Microsoft Phi
	"microsoft/phi-4-mini-flash-reasoning",
	"microsoft/phi-4-mini-instruct",
	// IBM
	"ibm/granite-3.3-8b-instruct",
];

// =============================================================================
// Custom streaming - wraps standard openai-completions with NIM-specific fixes
// =============================================================================

/**
 * Custom streamSimple that wraps the standard OpenAI completions streamer.
 *
 * Fixes for NVIDIA NIM:
 * 1. Maps pi's thinking levels to values accepted by NVIDIA NIM
 * 2. Strips reasoning_effort for models where it doesn't trigger thinking
 * 3. Injects chat_template_kwargs per model to actually enable thinking
 * 4. Uses onPayload callback to mutate request params before they're sent
 */
type NimApiKeyEnvName = (typeof NVIDIA_API_KEY_ENV_NAMES)[number];
type AuthStorageLike = {
	get?: (provider: string) => unknown;
};

interface NimApiKeyCredential {
	type: "api_key";
	key: string;
}

function getNimApiKeyEnv(): NimApiKeyEnvName | undefined {
	return NVIDIA_API_KEY_ENV_NAMES.find((envName) => !!process.env[envName]);
}

function getNimApiKey(): string | undefined {
	const envName = getNimApiKeyEnv();
	const apiKey = envName ? process.env[envName] : undefined;
	return apiKey?.trim() || undefined;
}

function getNimProviderApiKeyConfig(): string {
	return `$${getNimApiKeyEnv() ?? NVIDIA_NIM_API_KEY_ENV}`;
}

function isNimApiKeyEnvName(value: string): value is NimApiKeyEnvName {
	return NVIDIA_API_KEY_ENV_NAMES.includes(value as NimApiKeyEnvName);
}

function isNimApiKeyEnvReference(value: string): boolean {
	return value.startsWith("$") && isNimApiKeyEnvName(value.slice(1));
}

function isNimApiKeyEnvPlaceholder(value: string): boolean {
	return isNimApiKeyEnvName(value) || isNimApiKeyEnvReference(value);
}

function resolveNimApiKeyEnvReference(value: string): string | undefined {
	if (!isNimApiKeyEnvReference(value)) return undefined;

	const envValue = process.env[value.slice(1)]?.trim();
	return envValue || undefined;
}

function isNimApiKeyEnvValue(value: string): boolean {
	return NVIDIA_API_KEY_ENV_NAMES.some((envName) => process.env[envName]?.trim() === value);
}

function isNimApiKeyCredential(credential: unknown): credential is NimApiKeyCredential {
	return (
		typeof credential === "object" &&
		credential !== null &&
		(credential as { type?: unknown }).type === "api_key" &&
		typeof (credential as { key?: unknown }).key === "string"
	);
}

function readStoredNimApiKeyConfig(): string | undefined {
	try {
		const authPath = join(getAgentDir(), "auth.json");
		if (!existsSync(authPath)) return undefined;

		const data = JSON.parse(readFileSync(authPath, "utf-8")) as Record<string, unknown>;
		const credential = data[PROVIDER_NAME];
		return isNimApiKeyCredential(credential) ? credential.key : undefined;
	} catch {
		return undefined;
	}
}

function getStoredNimApiKeyConfig(authStorage?: AuthStorageLike): string | undefined {
	if (authStorage) {
		try {
			const credential = authStorage.get?.(PROVIDER_NAME);
			return isNimApiKeyCredential(credential) ? credential.key : undefined;
		} catch {
			return undefined;
		}
	}

	return readStoredNimApiKeyConfig();
}

function hasStoredNimCommandCredential(authStorage?: AuthStorageLike): boolean {
	return getStoredNimApiKeyConfig(authStorage)?.startsWith("!") ?? false;
}

function getStoredResolvedNimApiKey(authStorage?: AuthStorageLike): string | undefined {
	const configuredApiKey = getStoredNimApiKeyConfig(authStorage)?.trim();
	if (!configuredApiKey || configuredApiKey.startsWith("!")) return undefined;

	const envValue = process.env[configuredApiKey]?.trim();
	return envValue || configuredApiKey;
}

function normalizeResolvedNimApiKey(apiKey: string | undefined): string | undefined {
	if (apiKey === undefined) return undefined;

	const trimmed = apiKey.trim();
	if (!trimmed) {
		throw new Error("NVIDIA NIM API key resolved to an empty value.");
	}

	return trimmed;
}

function resolveNimApiKey(apiKey: string | undefined, authStorage?: AuthStorageLike): string | undefined {
	const resolvedApiKey = normalizeResolvedNimApiKey(apiKey);
	const hasStoredCommandCredential = hasStoredNimCommandCredential(authStorage);

	if (
		hasStoredCommandCredential &&
		(!resolvedApiKey || isNimApiKeyEnvPlaceholder(resolvedApiKey) || isNimApiKeyEnvValue(resolvedApiKey))
	) {
		throw new Error("NVIDIA NIM API key command resolved to an empty value.");
	}

	const storedApiKey = getStoredResolvedNimApiKey(authStorage);
	if (storedApiKey && (!resolvedApiKey || isNimApiKeyEnvPlaceholder(resolvedApiKey))) return storedApiKey;

	if (resolvedApiKey) {
		const envReferenceApiKey = resolveNimApiKeyEnvReference(resolvedApiKey);
		if (envReferenceApiKey) return envReferenceApiKey;
		if (!isNimApiKeyEnvPlaceholder(resolvedApiKey)) return resolvedApiKey;
	}

	return getNimApiKey();
}

function resolveRequiredNimApiKey(apiKey: string | undefined): string {
	const resolvedApiKey = resolveNimApiKey(apiKey);
	if (resolvedApiKey) return resolvedApiKey;

	throw new Error(
		`NVIDIA NIM: no API key configured. Set ${NVIDIA_NIM_API_KEY_ENV} or ${NVIDIA_API_KEY_ENV}. ` +
		`Get a free API key at https://build.nvidia.com and export it: ` +
		`export ${NVIDIA_NIM_API_KEY_ENV}=nvapi-...`,
	);
}

function mapNimTopLevelReasoning(reasoning: SimpleStreamOptions["reasoning"]): SimpleStreamOptions["reasoning"] {
	if (reasoning === "minimal") return "low";
	if (reasoning === "xhigh") return "high";
	return reasoning;
}

function mapDeepSeekV4Reasoning(reasoning: SimpleStreamOptions["reasoning"]): "high" | "max" {
	return reasoning === "xhigh" ? "max" : "high";
}

function buildThinkingKwargs(
	thinkingConfig: ThinkingConfig,
	reasoning: SimpleStreamOptions["reasoning"],
): Record<string, unknown> {
	const kwargs = { ...thinkingConfig.enableKwargs };
	if (thinkingConfig.includeReasoningEffortInKwargs) {
		kwargs.reasoning_effort = mapDeepSeekV4Reasoning(reasoning);
	}
	return kwargs;
}

function buildNimRequestHeaders(headers: SimpleStreamOptions["headers"], apiKey: string): Record<string, string> {
	const resolvedHeaders: Record<string, string> = {};

	for (const [key, value] of Object.entries(headers ?? {})) {
		if (key.toLowerCase() === "authorization") continue;
		resolvedHeaders[key] = value;
	}

	return {
		...resolvedHeaders,
		Authorization: `Bearer ${apiKey}`,
	};
}

function nimStreamSimple(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	// pi-coding-agent registers streamSimple globally per `api` type (not per provider).
	// This streamer is invoked for ALL openai-completions providers (e.g. openrouter, openai),
	// not just nvidia-nim. Pass non-NIM calls through unchanged so we don't leak the
	// NVIDIA_NIM_API_KEY into other providers' Authorization headers.
	if (model.provider !== PROVIDER_NAME) {
		return streamSimpleOpenAICompletions(model as Model<"openai-completions">, context, options);
	}

	const thinkingConfig = THINKING_CONFIGS[model.id];
	const reasoning = options?.reasoning;
	const isThinkingEnabled = !!reasoning;

	// Map provider-agnostic pi levels to NIM's accepted top-level values.
	// Model-specific chat_template_kwargs may apply a different mapping below.
	const mappedReasoning = mapNimTopLevelReasoning(reasoning);

	// For models that have a thinking config: we handle thinking via chat_template_kwargs.
	// Suppress reasoning_effort (set reasoning to undefined) unless the model explicitly
	// supports it alongside chat_template_kwargs (like Kimi).
	let effectiveReasoning = mappedReasoning;
	if (thinkingConfig && isThinkingEnabled && !thinkingConfig.sendReasoningEffort) {
		// Don't send reasoning_effort - we'll use chat_template_kwargs instead.
		// Setting to undefined prevents buildParams from adding reasoning_effort.
		effectiveReasoning = undefined;
	}

	// Use pi's already-resolved provider key when available (auth.json, shell command,
	// CLI override), and fall back to the two NVIDIA environment variable names.
	const nimApiKey = resolveRequiredNimApiKey(options?.apiKey);

	const modifiedOptions: SimpleStreamOptions = {
		...options,
		reasoning: effectiveReasoning,
		apiKey: nimApiKey,
		headers: buildNimRequestHeaders(options?.headers, nimApiKey),
		onPayload: (params: unknown) => {
			const p = params as Record<string, unknown>;

			if (thinkingConfig) {
				if (isThinkingEnabled) {
					// Inject chat_template_kwargs to enable thinking
					p.chat_template_kwargs = buildThinkingKwargs(thinkingConfig, reasoning);
				} else if (thinkingConfig.disableKwargs) {
					// Explicitly disable thinking (some models think by default, e.g. GLM-5.1/5/4.7)
					p.chat_template_kwargs = thinkingConfig.disableKwargs;
				}
			}

			// Ensure reasoning_effort is never "minimal" (belt & suspenders)
			if (p.reasoning_effort === "minimal") {
				p.reasoning_effort = "low";
			}

			// Normalize content arrays to plain strings where possible.
			// Many older/smaller NIM models (e.g., solar, baichuan, falcon) reject the
			// array format [{"type":"text","text":"..."}] and require a plain string.
			// This is safe for all models since plain strings are universally accepted.
			const messages = p.messages as Array<Record<string, unknown>> | undefined;
			if (messages) {
				for (const msg of messages) {
					if (Array.isArray(msg.content)) {
						const parts = msg.content as Array<Record<string, unknown>>;
						const allText = parts.every((part) => part.type === "text");
						if (allText) {
							msg.content = parts.map((part) => part.text as string).join("\n");
						}
					}
				}
			}

			// Chain to original onPayload if present
			return options?.onPayload?.(params, model);
		},
	};

	return streamSimpleOpenAICompletions(model as Model<"openai-completions">, context, modifiedOptions);
}

// =============================================================================
// Model building helpers
// =============================================================================

interface NimModelEntry {
	id: string;
	name: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	contextWindow: number;
	maxTokens: number;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
	compat?: Record<string, unknown>;
}

function makeDisplayName(modelId: string): string {
	const parts = modelId.split("/");
	const name = parts[parts.length - 1];
	return name
		.replace(/-/g, " ")
		.replace(/_/g, " ")
		.replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildModelEntry(modelId: string): NimModelEntry | null {
	if (SKIP_MODELS.has(modelId)) return null;

	const isReasoning = REASONING_MODELS.has(modelId);
	const isVision = VISION_MODELS.has(modelId);
	const contextWindow = CONTEXT_WINDOWS[modelId] ?? 4096;
	const maxTokens = MAX_TOKENS[modelId] ?? Math.min(2048, contextWindow);

	const entry: NimModelEntry = {
		id: modelId,
		name: makeDisplayName(modelId),
		reasoning: isReasoning,
		input: isVision ? ["text", "image"] : ["text"],
		contextWindow,
		maxTokens,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	};

	// Default compat for all NIM models:
	// - supportsReasoningEffort: false - we handle thinking via streamSimple + chat_template_kwargs
	// - supportsDeveloperRole: false - "developer" role + chat_template_kwargs causes 500 on NIM
	//   (developer role alone works, but combined with thinking kwargs it breaks)
	// - maxTokensField: "max_tokens" - safer default for heterogeneous backends
	entry.compat = {
		supportsReasoningEffort: false,
		supportsDeveloperRole: false,
		maxTokensField: "max_tokens",
	};

	// Mistral models on NIM need extra compat flags
	if (modelId.startsWith("mistralai/")) {
		entry.compat.requiresToolResultName = true;
		entry.compat.requiresThinkingAsText = true;
		entry.compat.requiresMistralToolIds = true;
	}

	return entry;
}

// =============================================================================
// Dynamic model discovery
// =============================================================================

interface NimApiModel {
	id: string;
	object: string;
	owned_by: string;
}

type NimModelFetchResult =
	| { ok: true; modelIds: string[] }
	| { ok: false; reason: "auth" | "transient" | "invalid" | "network" | "other" };

const NIM_DISCOVERY_CREDENTIAL_WARNING =
	"NVIDIA NIM model discovery skipped: check your nvidia-nim credentials.";

function sanitizeNimLogMessage(message: string): string {
	return message.replace(/nvapi-[A-Za-z0-9._-]+/g, "nvapi-[REDACTED]");
}

function notifyNimDiscoveryCredentialWarning(ctx: any): void {
	ctx?.ui?.notify?.(NIM_DISCOVERY_CREDENTIAL_WARNING, "warning");
}

async function resolveNimDiscoveryApiKey(ctx: any): Promise<string | undefined> {
	try {
		const apiKey = await ctx?.modelRegistry?.getApiKeyForProvider?.(PROVIDER_NAME);
		return resolveNimApiKey(apiKey, ctx?.modelRegistry?.authStorage);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		console.warn(`pi-nvidia-nim: ${sanitizeNimLogMessage(message)}`);
		notifyNimDiscoveryCredentialWarning(ctx);
		return undefined;
	}
}

async function fetchNimModels(apiKey: string): Promise<NimModelFetchResult> {
	try {
		const response = await fetch(`${NVIDIA_NIM_BASE_URL}/models`, {
			headers: {
				Authorization: `Bearer ${apiKey}`,
				Accept: "application/json",
			},
			signal: AbortSignal.timeout(10000),
		});

		if (response.status === 401 || response.status === 403) {
			return { ok: false, reason: "auth" };
		}

		if (response.status === 429 || response.status >= 500) {
			return { ok: false, reason: "transient" };
		}

		if (!response.ok) return { ok: false, reason: "other" };

		const data = (await response.json()) as { data?: NimApiModel[] };
		if (!Array.isArray(data.data)) return { ok: false, reason: "invalid" };

		return {
			ok: true,
			modelIds: data.data.map((m) => m.id).filter((id): id is string => typeof id === "string" && id.length > 0),
		};
	} catch {
		return { ok: false, reason: "network" };
	}
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
	const providerApiKeyConfig = getNimProviderApiKeyConfig();

	// Always register the curated model list. The request path resolves credentials
	// through pi first (CLI override, auth.json, shell command), then falls back to
	// NVIDIA_NIM_API_KEY/NVIDIA_API_KEY. This keeps models available even when pi
	// was launched by a shell that did not source ~/.bashrc or ~/.zshrc.

	// Build the curated model list
	const modelMap = new Map<string, NimModelEntry>();

	// Add featured models first (preserves order in selector)
	for (const id of FEATURED_MODELS) {
		const entry = buildModelEntry(id);
		if (entry) modelMap.set(id, entry);
	}

	// Register with curated models immediately
	const curatedModels = Array.from(modelMap.values());

	pi.registerProvider(PROVIDER_NAME, {
		baseUrl: NVIDIA_NIM_BASE_URL,
		apiKey: providerApiKeyConfig,
		api: "openai-completions",
		authHeader: true,
		models: curatedModels,
		streamSimple: nimStreamSimple,
	});

	// On session start, discover additional models from the API
	pi.on("session_start", async (_event: any, ctx: any) => {
		const apiKey = await resolveNimDiscoveryApiKey(ctx);
		if (!apiKey) return;

		// Fetch live model list
		const fetchResult = await fetchNimModels(apiKey);
		if (!fetchResult.ok) {
			if (fetchResult.reason === "auth") {
				notifyNimDiscoveryCredentialWarning(ctx);
			}
			return;
		}

		const liveModelIds = fetchResult.modelIds;
		if (liveModelIds.length === 0) return;

		let newModelsAdded = 0;
		for (const id of liveModelIds) {
			if (modelMap.has(id)) continue;
			const entry = buildModelEntry(id);
			if (entry) {
				modelMap.set(id, entry);
				newModelsAdded++;
			}
		}

		// Re-register with full model list if we found new ones.
		// NOTE: must use ctx.modelRegistry.registerProvider() here, not pi.registerProvider().
		// pi.registerProvider() only queues registrations for the initial extension load.
		// From event handlers/commands, we need to call the registry directly.
		if (newModelsAdded > 0) {
			const allModels = Array.from(modelMap.values());
			ctx.modelRegistry.registerProvider(PROVIDER_NAME, {
				baseUrl: NVIDIA_NIM_BASE_URL,
				apiKey: getNimProviderApiKeyConfig(),
				api: "openai-completions",
				authHeader: true,
				models: allModels,
				streamSimple: nimStreamSimple,
			});
		}
	});

}
