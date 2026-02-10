/**
 * NVIDIA NIM API Provider Extension for pi
 *
 * Provides access to 100+ models from NVIDIA's NIM platform (build.nvidia.com)
 * via their OpenAI-compatible API endpoint.
 *
 * Setup:
 *   1. Get an API key from https://build.nvidia.com
 *   2. Export it: export NVIDIA_NIM_API_KEY=nvapi-...
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
 * - GLM-4.7:       `chat_template_kwargs: { enable_thinking: true, clear_thinking: false }`
 * - Kimi K2.5:     `chat_template_kwargs: { thinking: true }` (also accepts reasoning_effort)
 * - Qwen3:         `chat_template_kwargs: { enable_thinking: true }`
 *
 * NIM only accepts `reasoning_effort` values of "low", "medium", "high" — NOT
 * "minimal". The extension maps pi's "minimal" level to "low" automatically.
 *
 * Some models (e.g., GLM-4.7) always produce reasoning output regardless of
 * thinking settings.
 */

import type {
	Api,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
} from "@mariozechner/pi-ai";
import { streamSimpleOpenAICompletions } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

// =============================================================================
// Constants
// =============================================================================

const NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1";
const NVIDIA_NIM_API_KEY_ENV = "NVIDIA_NIM_API_KEY";
const PROVIDER_NAME = "nvidia-nim";

// =============================================================================
// Per-model thinking configuration
// =============================================================================

/**
 * Maps model ID prefixes/exact IDs to their chat_template_kwargs for thinking.
 * When a user enables thinking in pi (any level > off), we inject these kwargs
 * into the request body. Models not listed here either:
 * - Don't support thinking (non-reasoning models)
 * - Always think regardless (GLM-4.7 without explicit kwargs)
 * - Work with standard reasoning_effort (rare on NIM)
 */
interface ThinkingConfig {
	/** chat_template_kwargs to send when thinking is enabled */
	enableKwargs: Record<string, unknown>;
	/** chat_template_kwargs to send when thinking is explicitly disabled (optional) */
	disableKwargs?: Record<string, unknown>;
	/** If true, also send reasoning_effort alongside chat_template_kwargs */
	sendReasoningEffort?: boolean;
}

const THINKING_CONFIGS: Record<string, ThinkingConfig> = {
	// DeepSeek models need chat_template_kwargs — reasoning_effort alone doesn't trigger thinking
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
	// GLM-4.7 always thinks by default, but can be controlled
	"z-ai/glm4.7": {
		enableKwargs: { enable_thinking: true, clear_thinking: false },
		disableKwargs: { enable_thinking: false },
	},
	// Kimi models: chat_template_kwargs works, reasoning_effort also works
	"moonshotai/kimi-k2.5": {
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
	"deepseek-ai/deepseek-r1-distill-llama-8b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-14b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-32b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-7b": 131072,
	"deepseek-ai/deepseek-coder-6.7b-instruct": 16384,
	// Kimi / Moonshot
	"moonshotai/kimi-k2-instruct": 131072,
	"moonshotai/kimi-k2-instruct-0905": 131072,
	"moonshotai/kimi-k2-thinking": 131072,
	"moonshotai/kimi-k2.5": 262144,
	// MiniMax
	"minimaxai/minimax-m2": 1048576,
	"minimaxai/minimax-m2.1": 1048576,
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
};

// Known max output tokens
const MAX_TOKENS: Record<string, number> = {
	"deepseek-ai/deepseek-v3.1": 16384,
	"deepseek-ai/deepseek-v3.1-terminus": 16384,
	"deepseek-ai/deepseek-v3.2": 16384,
	"moonshotai/kimi-k2.5": 16384,
	"moonshotai/kimi-k2-instruct": 8192,
	"moonshotai/kimi-k2-thinking": 16384,
	"minimaxai/minimax-m2": 8192,
	"minimaxai/minimax-m2.1": 8192,
	"meta/llama-4-maverick-17b-128e-instruct": 16384,
	"meta/llama-4-scout-17b-16e-instruct": 16384,
	"z-ai/glm4.7": 16384,
	"qwen/qwen3-coder-480b-a35b-instruct": 65536,
	"nvidia/llama-3.1-nemotron-ultra-253b-v1": 32768,
	"openai/gpt-oss-120b": 16384,
	"openai/gpt-oss-20b": 16384,
	"mistralai/mistral-large-3-675b-instruct-2512": 16384,
	"mistralai/devstral-2-123b-instruct-2512": 32768,
};

// =============================================================================
// Curated "featured" models — listed first in the model selector
// =============================================================================

const FEATURED_MODELS = [
	// Flagship / frontier
	"deepseek-ai/deepseek-v3.2",
	"deepseek-ai/deepseek-v3.1",
	"deepseek-ai/deepseek-v3.1-terminus",
	"moonshotai/kimi-k2.5",
	"moonshotai/kimi-k2-thinking",
	"moonshotai/kimi-k2-instruct",
	"moonshotai/kimi-k2-instruct-0905",
	"minimaxai/minimax-m2.1",
	"minimaxai/minimax-m2",
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
// Custom streaming — wraps standard openai-completions with NIM-specific fixes
// =============================================================================

/**
 * Custom streamSimple that wraps the standard OpenAI completions streamer.
 *
 * Fixes for NVIDIA NIM:
 * 1. Maps pi's "minimal" thinking level → "low" (NIM only accepts low/medium/high)
 * 2. Strips reasoning_effort for models where it doesn't trigger thinking
 * 3. Injects chat_template_kwargs per model to actually enable thinking
 * 4. Uses onPayload callback to mutate request params before they're sent
 */
function nimStreamSimple(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const thinkingConfig = THINKING_CONFIGS[model.id];
	const reasoning = options?.reasoning;
	const isThinkingEnabled = reasoning && reasoning !== "off";

	// Map "minimal" → "low" since NIM rejects "minimal" with a 400 error.
	// NIM only accepts: "low", "medium", "high"
	let mappedReasoning = reasoning;
	if (reasoning === "minimal") {
		mappedReasoning = "low";
	}

	// For models that have a thinking config: we handle thinking via chat_template_kwargs.
	// Suppress reasoning_effort (set reasoning to undefined) unless the model explicitly
	// supports it alongside chat_template_kwargs (like Kimi).
	let effectiveReasoning = mappedReasoning;
	if (thinkingConfig && isThinkingEnabled && !thinkingConfig.sendReasoningEffort) {
		// Don't send reasoning_effort — we'll use chat_template_kwargs instead.
		// Setting to undefined prevents buildParams from adding reasoning_effort.
		effectiveReasoning = undefined;
	}

	const modifiedOptions: SimpleStreamOptions = {
		...options,
		reasoning: effectiveReasoning,
		onPayload: (params: unknown) => {
			const p = params as Record<string, unknown>;

			if (thinkingConfig) {
				if (isThinkingEnabled) {
					// Inject chat_template_kwargs to enable thinking
					p.chat_template_kwargs = thinkingConfig.enableKwargs;
				} else if (thinkingConfig.disableKwargs) {
					// Explicitly disable thinking (some models think by default, e.g. GLM-4.7)
					p.chat_template_kwargs = thinkingConfig.disableKwargs;
				}
			}

			// Ensure reasoning_effort is never "minimal" (belt & suspenders)
			if (p.reasoning_effort === "minimal") {
				p.reasoning_effort = "low";
			}

			// Chain to original onPayload if present
			options?.onPayload?.(params);
		},
	};

	return streamSimpleOpenAICompletions(model, context, modifiedOptions);
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
	const contextWindow = CONTEXT_WINDOWS[modelId] ?? 32768;
	const maxTokens = MAX_TOKENS[modelId] ?? 8192;

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
	// - supportsReasoningEffort: false — we handle thinking via streamSimple + chat_template_kwargs
	// - supportsDeveloperRole: false — "developer" role + chat_template_kwargs causes 500 on NIM
	//   (developer role alone works, but combined with thinking kwargs it breaks)
	// - maxTokensField: "max_tokens" — safer default for heterogeneous backends
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

async function fetchNimModels(apiKey: string): Promise<string[]> {
	try {
		const response = await fetch(`${NVIDIA_NIM_BASE_URL}/models`, {
			headers: {
				Authorization: `Bearer ${apiKey}`,
				Accept: "application/json",
			},
			signal: AbortSignal.timeout(10000),
		});

		if (!response.ok) return [];

		const data = (await response.json()) as { data: NimApiModel[] };
		return data.data?.map((m) => m.id) ?? [];
	} catch {
		return [];
	}
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
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
		apiKey: NVIDIA_NIM_API_KEY_ENV,
		api: "openai-completions",
		authHeader: true,
		models: curatedModels,
		streamSimple: nimStreamSimple,
	});

	// On session start, discover additional models from the API
	pi.on("session_start", async (_event, ctx) => {
		const apiKey = process.env[NVIDIA_NIM_API_KEY_ENV];
		if (!apiKey) {
			ctx.ui.notify(
				`NVIDIA NIM: Set ${NVIDIA_NIM_API_KEY_ENV} env var to enable models. Get a key at https://build.nvidia.com`,
				"warning",
			);
			return;
		}

		// Fetch live model list
		const liveModelIds = await fetchNimModels(apiKey);
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

		// Re-register with full model list if we found new ones
		if (newModelsAdded > 0) {
			const allModels = Array.from(modelMap.values());
			pi.registerProvider(PROVIDER_NAME, {
				baseUrl: NVIDIA_NIM_BASE_URL,
				apiKey: NVIDIA_NIM_API_KEY_ENV,
				api: "openai-completions",
				authHeader: true,
				models: allModels,
				streamSimple: nimStreamSimple,
			});
		}
	});

	// Register a command to browse and switch to NIM models
	pi.registerCommand("nim-models", {
		description: "Browse and switch to an NVIDIA NIM model",
		handler: async (_args, ctx) => {
			const models = Array.from(modelMap.values());
			if (models.length === 0) {
				ctx.ui.notify("No NVIDIA NIM models loaded. Set NVIDIA_NIM_API_KEY.", "warning");
				return;
			}

			const lines = models.map((m) => {
				const tags: string[] = [];
				if (m.reasoning) tags.push("reasoning");
				if (m.input.includes("image")) tags.push("vision");
				const tagStr = tags.length > 0 ? ` [${tags.join(", ")}]` : "";
				return `${m.id}${tagStr} (ctx: ${(m.contextWindow / 1024).toFixed(0)}k)`;
			});

			const choice = await ctx.ui.select(`NVIDIA NIM Models (${models.length})`, lines);

			if (choice !== undefined) {
				const model = models[choice];
				const fullModel = ctx.modelRegistry.find(PROVIDER_NAME, model.id);
				if (fullModel) {
					const success = await pi.setModel(fullModel);
					if (success) {
						ctx.ui.notify(`Switched to ${model.id}`, "info");
					} else {
						ctx.ui.notify(`Failed to switch — check your API key`, "error");
					}
				}
			}
		},
	});

	// Register a command to refresh models from the API
	pi.registerCommand("nim-refresh", {
		description: "Refresh NVIDIA NIM model list from API",
		handler: async (_args, ctx) => {
			const apiKey = process.env[NVIDIA_NIM_API_KEY_ENV];
			if (!apiKey) {
				ctx.ui.notify(`Set ${NVIDIA_NIM_API_KEY_ENV} env var first`, "error");
				return;
			}

			ctx.ui.notify("Fetching models from NVIDIA NIM API...", "info");
			const liveModelIds = await fetchNimModels(apiKey);

			if (liveModelIds.length === 0) {
				ctx.ui.notify("Could not fetch models. Check your API key.", "error");
				return;
			}

			let newCount = 0;
			for (const id of liveModelIds) {
				if (modelMap.has(id)) continue;
				const entry = buildModelEntry(id);
				if (entry) {
					modelMap.set(id, entry);
					newCount++;
				}
			}

			const allModels = Array.from(modelMap.values());
			pi.registerProvider(PROVIDER_NAME, {
				baseUrl: NVIDIA_NIM_BASE_URL,
				apiKey: NVIDIA_NIM_API_KEY_ENV,
				api: "openai-completions",
				authHeader: true,
				models: allModels,
				streamSimple: nimStreamSimple,
			});

			ctx.ui.notify(
				`NVIDIA NIM: ${allModels.length} total models (${newCount} newly discovered)`,
				"info",
			);
		},
	});
}
